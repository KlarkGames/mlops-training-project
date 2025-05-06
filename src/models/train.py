import warnings

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from src.data.dataloader import ASRDataset, collate_fn
from src.models.model import ASRLightningConformer

warnings.filterwarnings("ignore", category=FutureWarning)


@click.command()
@click.option("--train-csv", envvar="TRAIN_CSV", show_envvar=True, help="Path to training CSV file")
@click.option("--val-csv", envvar="VAL_CSV", show_envvar=True, help="Path to development CSV file")
@click.option(
    "--train-audio-dir", envvar="TRAIN_AUDIO_DIR", show_envvar=True, help="Directory with training audio files"
)
@click.option(
    "--val-audio-dir", envvar="VAL_AUDIO_DIR", show_envvar=True, help="Directory with development audio files"
)
@click.option(
    "--vocabulary-json",
    envvar="VOCABULARY_JSON",
    show_envvar=True,
    help="Path to JSON file with vocabulary list in it.",
)
# Model parameters
@click.option(
    "--input-dim",
    envvar="INPUT_DIM",
    type=int,
    default=80,
    show_envvar=True,
    help="Number of mel bins (input dimension)",
)
@click.option(
    "--num-heads", envvar="NUM_HEADS", type=int, default=8, show_envvar=True, help="Number of attention heads"
)
@click.option(
    "--ffn-dim", envvar="FFN_DIM", type=int, default=1024, show_envvar=True, help="Dimension of feed-forward network"
)
@click.option(
    "--num-layers", envvar="NUM_LAYERS", type=int, default=6, show_envvar=True, help="Number of conformer layers"
)
@click.option(
    "--kernel-size",
    envvar="KERNEL_SIZE",
    type=int,
    default=31,
    show_envvar=True,
    help="Depthwise convolution kernel size",
)
@click.option("--dropout", envvar="DROPOUT", type=float, default=0.1, show_envvar=True, help="Dropout probability")
@click.option(
    "--subsampling-factor",
    envvar="SUBSAMPLING_FACTOR",
    type=int,
    default=4,
    show_envvar=True,
    help="Subsampling factor",
)
# Optimizer and scheduler
@click.option("--lr", envvar="LR", type=float, default=0.001, show_envvar=True, help="Learning rate")
@click.option(
    "--weight-decay",
    envvar="WEIGHT_DECAY",
    type=float,
    default=1e-4,
    show_envvar=True,
    help="Weight decay for optimizer",
)
@click.option(
    "--scheduler-t0",
    envvar="SCHEDULER_T0",
    type=int,
    default=10,
    show_envvar=True,
    help="Scheduler t_0 for CosineAnnealingWarmRestarts",
)
@click.option(
    "--scheduler-t-mult",
    envvar="SCHEDULER_T_MULT",
    type=int,
    default=1,
    show_envvar=True,
    help="Scheduler t_mult for CosineAnnealingWarmRestarts",
)
@click.option(
    "--scheduler-eta-min",
    envvar="SCHEDULER_ETA_MIN",
    type=float,
    default=5e-6,
    show_envvar=True,
    help="Minimum learning rate for scheduler",
)
# Training
@click.option(
    "--batch-size", envvar="BATCH_SIZE", type=int, default=8, show_envvar=True, help="Batch size for training"
)
@click.option(
    "--max-epochs", envvar="MAX_EPOCHS", type=int, default=50, show_envvar=True, help="Maximum number of epochs"
)
@click.option(
    "--es-patience", envvar="ES_PATIENCE", type=int, default=65, show_envvar=True, help="Early stopping patience"
)
@click.option(
    "--log-every-n-steps",
    envvar="LOG_EVERY_N_STEPS",
    type=int,
    default=10,
    show_envvar=True,
    help="How often to log training steps",
)
@click.option(
    "--ckpt-path", envvar="CKPT_PATH", default=None, show_envvar=True, help="Path to checkpoint to resume training"
)
def main(
    train_csv,
    val_csv,
    train_audio_dir,
    val_audio_dir,
    vocabulary_json,
    input_dim,
    num_heads,
    ffn_dim,
    num_layers,
    kernel_size,
    dropout,
    subsampling_factor,
    lr,
    weight_decay,
    scheduler_t0,
    scheduler_t_mult,
    scheduler_eta_min,
    batch_size,
    max_epochs,
    es_patience,
    log_every_n_steps,
    ckpt_path,
):
    mlflow_logger = MLFlowLogger(experiment_name="ASR_Conformer_Experiment", tracking_uri="file:./mlruns")

    train_ds = ASRDataset(train_csv, train_audio_dir, vocabulary_json, target_sample_rate=16000)
    dev_ds = ASRDataset(val_csv, val_audio_dir, vocabulary_json, target_sample_rate=16000)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn)

    early_stop = EarlyStopping(monitor="val_loss", patience=es_patience, mode="min", verbose=True)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Model setup
    model = ASRLightningConformer(
        input_dim=input_dim,
        num_classes=train_ds.num_classes,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        depthwise_conv_kernel_size=kernel_size,
        dropout=dropout,
        learning_rate=lr,
        weight_decay=weight_decay,
        subsampling_factor=subsampling_factor,
        scheduler_t_0=scheduler_t0,
        scheduler_t_mult=scheduler_t_mult,
        scheduler_min_lr=scheduler_eta_min,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stop, checkpoint, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=log_every_n_steps,
        logger=mlflow_logger,
    )

    # Training
    trainer.fit(model, train_loader, dev_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
