import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchaudio.models import Conformer

warnings.filterwarnings("ignore", category=FutureWarning)


class ASRLightningConformer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        subsampling_factor: int,
        scheduler_t_0: int,
        scheduler_t_mult: int,
        scheduler_min_lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.subsampling_factor = subsampling_factor

        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False,
        )
        self.fc = nn.Linear(input_dim, num_classes)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x, lengths):
        x, out_lens = self.conformer(x, lengths)
        x = self.fc(x)
        return x.transpose(0, 1), out_lens

    def training_step(self, batch, batch_idx):
        specs, lens, targets, t_lens = batch
        logits, out_lens = self(specs, lens)

        eff = out_lens // self.subsampling_factor
        eff = torch.clamp(eff, min=1)
        loss = self.criterion(logits, targets, eff, t_lens)
        self.log("train_loss", loss, prog_bar=True, batch_size=specs.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        specs, lens, targets, t_lens = batch
        logits, out_lens = self(specs, lens)

        eff = out_lens // self.subsampling_factor
        eff = torch.clamp(eff, min=1)
        loss = self.criterion(logits, targets, eff, t_lens)
        self.log("val_loss", loss, prog_bar=True, batch_size=specs.size(0))

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.scheduler_t_0,
            T_mult=self.hparams.scheduler_t_mult,
            eta_min=self.hparams.scheduler_min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
