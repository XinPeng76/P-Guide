import copy
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import torch
from absl import app, flags
from tqdm import trange
from diffusers import AutoencoderKL
from tqdm import tqdm
from utils_cifar import ema, infiniteloop,LabelToImage_ResNet


#from torchcfm.models.unet.unet import UNetModelWrapper
#from models.EDM import DhariwalUNet
from torchdyn.core import NeuralODE
from torchvision.utils import save_image
from models.DiT import *
# 🔧 MOD: 8-bit optimizer
import bitsandbytes as bnb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_F = LabelToImage_ResNet(num_classes=1000, img_shape=(4, 32, 32), base=64).to(device)

from torch.utils.data import Dataset
import torch
import glob

class LatentDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(folder + "/*.pt"))
        self.data = []

        for f in self.files:
            d = torch.load(f)
            self.data.append((d["z"], d["label"]))

        self.data = [(z[i], y[i]) for z, y in self.data for i in range(len(z))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def _encode_and_save(imgs, names, latent_dir, vae, scaling_factor):
    x = torch.stack(imgs).to(device)

    with torch.autocast("cuda"):
        z = vae.encode(x).latent_dist.sample()
        z = z * scaling_factor

    for zi, name in zip(z, names):
        torch.save(
            zi.cpu(),
            os.path.join(
                latent_dir,
                name.replace(".png", ".pt").replace(".jpg", ".pt"),
            ),
        )

# =====================================================
# Sampling (VAE + Latent FM)  ——  FP32 for stability
from torchdiffeq import odeint_adjoint as odeint
from functools import partial
def sample_from_model(model, x_0):
    t = torch.tensor([1.0, 0.0], dtype=x_0.dtype, device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
    return fake_image
# =====================================================
def generate_samples_vae(model, model_F, vae, parallel, savedir, step, net_="normal"):
    model.eval()
    model_ = copy.deepcopy(model).to(torch.float32).to(device)

    if parallel:
        model_ = model_.module.to(device)

    B = 16
    y = torch.randint(0, 1000, (B,), device="cuda")
    z = torch.randn(B, 4, 32, 32, device="cuda")
    mu, sigma = model_F(y)
    #xT = xT * 3
    x0 = mu + sigma * z
    #x0 = x0.to(torch.bfloat16).to(device)
    sample_model = partial(model_, y=y)

    with torch.no_grad():
        z = sample_from_model(sample_model, x0)[-1]
        #print(z.shape)
        z = z / 0.18215
        z = z.to(torch.float32)
        x = vae.decode(z).sample
        x = (x.clamp(-1, 1) + 1) / 2.0

    save_image(
        x,
        f"{savedir}/{net_}_generated_step_{step}.png",
        nrow=4,
    )

    model.train()


FLAGS = flags.FLAGS

# =====================================================
# Args
# =====================================================
flags.DEFINE_string("output_dir", "/root/autodl-tmp/results/", help="output directory")
flags.DEFINE_string(
    "pretrained_autoencoder_ckpt",
    "/root/autodl-tmp/stabilityai/sd-vae-ft-mse",
    help="VAE checkpoint",
)

# UNet
flags.DEFINE_integer("num_channel", 256, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient clip")
flags.DEFINE_integer("total_steps", 400001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="lr warmup steps")
flags.DEFINE_integer("batch_size", 32, help="batch size (4090: 32~64)")
flags.DEFINE_integer("num_workers", 8, help="dataloader workers")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay")
flags.DEFINE_bool("parallel", False, help="use DataParallel")

# Saving
flags.DEFINE_integer("save_step", 20000, help="save frequency")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Dataset
# =====================================================
def FlowMatching_loss(model, model_F, x, y,  device='cuda'):

    x = x.to(device)
    y = y.to(device)

    B = x.shape[0]

    z = torch.randn_like(x)

    mu, sigma = model_F(y)

    z = mu + sigma * z
    t = torch.rand(B, 1, device=device)
    t4 = t.view(B, 1, 1, 1).to(torch.bfloat16)


    x_t =  (1-t4) *x + t4 * z


    true_v = z - x
    x_t = x_t.to(torch.bfloat16)
    pred_v = model(t.squeeze(-1), x_t,y)

    return (true_v - pred_v).square().mean()




def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup
import torch.nn.functional as F
# =====================================================
# Train
# =====================================================
def train(argv):

    # ---------------------
    # VAE (frozen)
    # ---------------------
    vae = AutoencoderKL.from_pretrained(
        FLAGS.pretrained_autoencoder_ckpt
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    dataset = LatentDataset("/root/autodl-tmp/imagenet_latents")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    #datalooper = infiniteloop(dataloader)
    # ---------------------
    # DiT latent model
    # ---------------------
    net_model = DiT_B_2(img_resolution=32, in_channels=4, label_dropout=0.1, num_classes=1000)
    ckpt = torch.load("/root/autodl-tmp/model_875.pth",
    map_location=device,
)
    print("Finish loading model")
    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    net_model.load_state_dict(ckpt, strict=True)
    net_model.eval()
    ema_model = copy.deepcopy(net_model).to(device)

    if FLAGS.parallel:
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    model_F_size = 0
    for param in model_F.parameters():
        model_F_size += param.data.nelement()
    print("Model_F params: %.2f M" % (model_F_size / 1024 / 1024))
    # ---------------------
    # 🔧 MOD: AdamW 8-bit
    # ---------------------
    optim = bnb.optim.AdamW8bit(
        net_model.parameters(),
        lr=FLAGS.lr
    )
    optim2 = bnb.optim.AdamW8bit(model_F.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # 🔧 MOD: AMP scaler
    scaler = torch.GradScaler()

    # ---------------------
    # Flow Matcher
    # ---------------------

    savedir = os.path.join(FLAGS.output_dir, FLAGS.model)
    os.makedirs(savedir, exist_ok=True)
    dtype = torch.float32

    with trange(FLAGS.total_steps//10, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim2.zero_grad()
            batch = next(iter(dataloader))
            x1 = batch[0].to(device)
            y = batch[1].to(device)
            mu, sigma = model_F(y)
            logvar = 2 * torch.log(sigma + 1e-8)
            loss = (
                (x1 - mu).pow(2) / (2 * sigma.pow(2) + 1e-8)
                + 0.5 * logvar
            ).mean()
            loss.backward()
            optim2.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        model_F.eval()
    scaler = torch.GradScaler()
    net_model = net_model.to(torch.bfloat16).to(device)
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            batch = next(iter(dataloader))
            x1 = batch[0].to(device)
            #print("x1 shape:", x1.shape)
            y = batch[1].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = FlowMatching_loss(net_model, model_F, x1, y,  device='cuda')

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            #ema(net_model, ema_model, FLAGS.ema_decay)  # new
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            # sample and Saving the weights


            if step % FLAGS.save_step == 0:
                generate_samples_vae(
                    net_model,model_F, vae, FLAGS.parallel, savedir, step, "normal"
                )
                """
                generate_samples_vae(
                    ema_model,model_F, vae, FLAGS.parallel, savedir, step, "ema"
                )"""

                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        #"ema_model": ema_model.state_dict(),
                        "F_model": model_F.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    f"{savedir}/ckpt_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
