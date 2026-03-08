import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import torchvision
import numpy as np
from tqdm import tqdm

from models import utils as mutils
from models.ncsnpp import NCSNpp
import losses
import datasets
from configs.rectified_flow.zappos import get_config
import sde_lib
import sampling

parser = argparse.ArgumentParser(description="Training LoRA on Zappos dataset")
parser.add_argument(
    "--data_dir", 
    type=str, 
    required=True, 
    help="Chemin absolu ou relatif vers le dossier d'images Zappos"
)
args = parser.parse_args()

class ZapposDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = []
        
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, f))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        return {'image': self.transform(img)}

config = get_config()
device = torch.device("cuda")
batch_size = 1
lr = 1e-4

torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision('high')

score_model = mutils.create_model(config) 
if isinstance(score_model, torch.nn.DataParallel):
    score_model = score_model.module

ckpt = torch.load("logs/celebahq/checkpoints/checkpoint_10.pth", map_location=device, weights_only=False)
clean_state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
score_model.load_state_dict(clean_state_dict)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["Conv_0", "Conv_1", "Conv_2", "Dense_0"],
    lora_dropout=0.1, bias="none"
)
model = get_peft_model(score_model, lora_config)
model.to(device)

train_ds = ZapposDataset(data_dir=args.data_dir)
train_loader = DataLoader(
    train_ds, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
inverse_scaler = lambda x: (x + 1.) / 2.

sde = sde_lib.RectifiedFlow(
    init_type=config.sampling.init_type, 
    noise_scale=config.sampling.init_noise_scale, 
    use_ode_sampler="rk45", 
    sigma_var=config.sampling.sigma_variance, 
    ode_tol=config.sampling.ode_tol, 
    sample_N=config.sampling.sample_N
)

sampling_shape = (4, config.data.num_channels, config.data.image_size, config.data.image_size)
sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, eps=1e-3)

os.makedirs("temp", exist_ok=True)

eps = 1e-4
accumulation_steps = 8

for epoch in range(50):
    model.train() 
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
    
    for step, batch_data in enumerate(pbar):
        x1 = batch_data['image'].to(device, non_blocking=True).float()
        
        z0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device='cpu').to(device, non_blocking=True).view(-1, 1, 1, 1)
        t = t * (1.0 - eps) + eps
        
        xt = t * x1 + (1.0 - t) * z0
        target = x1 - z0
        
        prediction = model(xt, t.view(-1) * 999)
        
        loss = torch.mean(torch.square(prediction - target))
        loss_accum = loss / accumulation_steps
        loss_accum.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) 
        
        pbar.set_postfix({"Loss": f"{loss.item():.4e}"})
        
    torch.save(model.state_dict(), f"lora_zappos_last.pth")

    model.eval()
    with torch.no_grad():
        samples, _ = sampling_fn(model)
        samples = inverse_scaler(samples)
        samples = torch.clamp(samples, 0.0, 1.0)
        
        save_path = os.path.join("temp", f"sample_epoch_{epoch:02d}.png")
        torchvision.utils.save_image(samples, save_path, nrow=2, padding=2)
