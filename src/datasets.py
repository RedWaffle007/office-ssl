# src/datasets.py

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torch
import json
import pandas as pd

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

class MultiLabelSceneDataset(Dataset):
    """
    Dataset for multi-label scene images.
    Expects:
      images_root/ -> directory with images
      annotations_csv -> filename,labels (semicolon separated)
      label_map_json -> dict {class_name: index}
    Returns:
      image_tensor, target_vector (float tensor, size num_classes)
    """
    def __init__(self, images_root, annotations_csv, label_map_json, transform=None):
        self.images_root = Path(images_root)
        self.annotations_csv = Path(annotations_csv)
        self.label_map = json.load(open(label_map_json))
        self.class_names = {v: k for k, v in self.label_map.items()}

        df = pd.read_csv(self.annotations_csv)
        self.samples = []

        for _, row in df.iterrows():
            fname = str(row['filename'])
            labels = str(row['labels']) if not pd.isna(row['labels']) else ''
            labels = [s.strip() for s in labels.split(';') if s.strip()]
            img_path = self.images_root / fname
            if not img_path.exists():
                print(f" Missing image: {img_path}")
                continue

            target = torch.zeros(len(self.label_map), dtype=torch.float32)
            for lbl in labels:
                if lbl in self.label_map:
                    target[self.label_map[lbl]] = 1.0
                else:
                    print(f" Unknown label {lbl} in {fname}")
            self.samples.append((str(img_path), target))

        if transform is None:
            self.transform = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224, scale=(0.4, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        return img_t, target
