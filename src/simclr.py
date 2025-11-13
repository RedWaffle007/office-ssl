import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import random
import os
import pytorch_lightning as pl
from typing import List

# ---------- Dataset producing two views ----------
class TwoViewDataset(Dataset):
    def __init__(self, root: str, transform):
        self.root = Path(root)
        self.paths = [p for p in self.root.iterdir() if p.is_file()]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        return self.transform(img), self.transform(img)

# ---------- Model backbone + projection head ----------
class SimCLRModel(pl.LightningModule):
    def __init__(self, base_model='resnet50', proj_hidden_dim=2048, proj_out_dim=256, temperature=0.5, lr=1e-3, weight_decay=1e-6):
        super().__init__()
        self.save_hyperparameters()
        # backbone
        if base_model == 'resnet50':
            backbone = models.resnet50(pretrained=False)
            # remove fc
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            feat_dim = num_ftrs
        else:
            raise NotImplementedError("Only resnet50 supported in this script.")
        # projection head
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_out_dim)
        )
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        h = self.backbone(x)  # (B, feat_dim)
        z = self.projector(h)  # (B, out_dim)
        z = nn.functional.normalize(z, dim=1)
        return z

    def info_nce_loss(self, zis, zjs):
        # zis, zjs: (B, D)
        batch_size = zis.shape[0]
        z = torch.cat([zis, zjs], dim=0)  # 2B, D
        sim = torch.matmul(z, z.T)  # 2B x 2B
        sim = sim / self.temperature

        # mask to remove similarity with itself
        mask = (~torch.eye(2*batch_size, 2*batch_size, dtype=bool, device=sim.device)).float()
        # numerator: similarities of positive pairs
        positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
        nom = torch.exp(positives)

        # denominator
        denom = mask * torch.exp(sim)
        denom = denom.sum(dim=1)

        loss = -torch.log(nom / denom)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x1, x2 = batch  # each is shape (B, C, H, W)
        z1 = self(x1)
        z2 = self(x2)
        loss = self.info_nce_loss(z1, z2)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return [optimizer], [scheduler]

# ---------- Transforms ----------
def get_simclr_transform(img_size=224):
    # --- Robust GaussianBlur kernel calculation ---
    blur_kernel = max(3, int(round(0.1 * img_size)))
    # force it to be odd
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    # torchvision requires tuple of (k, k)
    kernel_tuple = (blur_kernel, blur_kernel)

    transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=kernel_tuple, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return transform


# ---------- Helper to build dataloader ----------
def build_dataloader(root, batch_size=64, num_workers=4, img_size=224):
    transform = get_simclr_transform(img_size)
    ds = TwoViewDataset(root, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dl
