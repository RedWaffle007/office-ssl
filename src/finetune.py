# src/finetune.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
import torchmetrics as tm

class MultiLabelFineTune(pl.LightningModule):
    """
    Fine-tune SimCLR backbone for multi-label classification (scene-level).
    """

    def __init__(self, backbone, num_classes, lr=1e-4, weight_decay=1e-6, freeze_backbone=False):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.backbone = backbone
        self.num_classes = num_classes

        # optionally freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feat_dim = self._get_feat_dim()
        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes)
        )

        self.criterion = nn.BCEWithLogitsLoss()

        # torchmetrics
        self.train_mAP = tm.classification.MultilabelAveragePrecision(num_labels=num_classes)
        self.val_mAP = tm.classification.MultilabelAveragePrecision(num_labels=num_classes)
        self.val_f1 = tm.classification.MultilabelF1Score(num_labels=num_classes, average='macro', threshold=0.5)

    def _get_feat_dim(self):
        """Get output dimension of backbone."""
        self.backbone.eval()
        device = next(self.backbone.parameters()).device
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224).to(device)
            out = self.backbone(x)
            if out.ndim == 4:
                out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1)).reshape(out.size(0), -1)
        return out.shape[1]

    def forward(self, x):
        f = self.backbone(x)
        if f.ndim == 4:
            f = torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)).reshape(f.size(0), -1)
        logits = self.classifier(f)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_mAP.update(preds, y.int())
        return loss

    def on_train_epoch_end(self):
        self.log('train/mAP', self.train_mAP.compute(), prog_bar=True)
        self.train_mAP.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits)
        self.val_mAP.update(preds, y.int())
        return {'val_loss': loss, 'preds': preds.detach().cpu(), 'targets': y.detach().cpu()}

    def validation_epoch_end(self, outputs):
        import numpy as np
        from sklearn.metrics import average_precision_score, f1_score, hamming_loss

        preds = torch.cat([o['preds'] for o in outputs], dim=0).numpy()
        targets = torch.cat([o['targets'] for o in outputs], dim=0).numpy()

        aps = []
        for i in range(targets.shape[1]):
            if targets[:, i].sum() == 0:
                aps.append(np.nan)
            else:
                aps.append(average_precision_score(targets[:, i], preds[:, i]))

        mean_ap = np.nanmean(aps)
        pred_bin = (preds >= 0.5).astype(int)
        f1 = f1_score(targets, pred_bin, average='macro', zero_division=0)
        ham = hamming_loss(targets, pred_bin)

        #  Ensure Lightning logs this metric on the epoch level
        self.log('val/mAP_mean', mean_ap, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log('val/f1_macro', f1, prog_bar=True, on_epoch=True)
        self.log('val/hamming_loss', ham, prog_bar=False, on_epoch=True)

        print(f"\nEpoch done — val_mAP_mean={mean_ap:.4f}, F1={f1:.4f}, Hamming={ham:.4f}")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return [opt], [sch]
