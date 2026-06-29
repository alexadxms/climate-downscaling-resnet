"""
Spatial downscaling of ERA5 2m temperature: 5.625deg -> 2.8125deg,
adapted from the ClimateLearn NeurIPS 2022 tutorial for the current
climate-learn API (https://github.com/aditya-grover/climate-learn).

Run as separate stages so a kill only costs you that one stage:
    source .venv/bin/activate
    python downscaling.py prepare     # download + convert (slow, one-off)
    python downscaling.py train       # train, checkpoints each epoch
    python downscaling.py train --resume   # continue from latest checkpoint
    python downscaling.py evaluate    # run trainer.test on the latest checkpoint
    python downscaling.py visualize   # produce outputs/*.png from the latest checkpoint
"""

import argparse
import glob
import os
import shutil
import time

import climate_learn as cl
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from climate_learn.data.processing.nc2npz import convert_nc2npz
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(__file__)
DATA_ROOT = os.path.join(PROJECT_DIR, ".climate_tutorial")
LOWRES_DIR = os.path.join(DATA_ROOT, "lowres")
HIGHRES_DIR = os.path.join(DATA_ROOT, "highres")
PROCESSED_LOW_DIR = os.path.join(DATA_ROOT, "processed_low")
PROCESSED_HIGH_DIR = os.path.join(DATA_ROOT, "processed_high")
VARIABLE = "2m_temperature"

# A single MPS forward+backward pass on this ResNet takes ~24s/batch (measured
# on this machine). The full 1979-2018 range at 6-hourly subsampling is ~3300
# batches/epoch (batch_size=16) -- about 22 hours per epoch. Restricted to a
# few years at daily subsampling, an epoch instead takes a few minutes.
TRAIN_START_YEAR = 1979
VAL_START_YEAR = 1980
TEST_START_YEAR = 1981
END_YEAR = 1982
SUBSAMPLE_HOURS = 24

CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")


def download():
    cl.data.download_weatherbench(
        dst=LOWRES_DIR, dataset="era5", variable=VARIABLE, resolution=5.625,
    )
    cl.data.download_weatherbench(
        dst=HIGHRES_DIR, dataset="era5", variable=VARIABLE, resolution=2.8125,
    )


def organize_into_variable_subfolders():
    # download_weatherbench saves .nc files flat into dst; convert_nc2npz
    # expects them under dst/<variable>/*.nc.
    for base in (LOWRES_DIR, HIGHRES_DIR):
        var_dir = os.path.join(base, VARIABLE)
        os.makedirs(var_dir, exist_ok=True)
        for f in os.listdir(base):
            src = os.path.join(base, f)
            if f.endswith(".nc") and os.path.isfile(src):
                shutil.move(src, os.path.join(var_dir, f))


def convert():
    for root_dir, save_dir in (
        (LOWRES_DIR, PROCESSED_LOW_DIR),
        (HIGHRES_DIR, PROCESSED_HIGH_DIR),
    ):
        convert_nc2npz(
            root_dir=root_dir,
            save_dir=save_dir,
            variables=[VARIABLE],
            start_train_year=TRAIN_START_YEAR,
            start_val_year=VAL_START_YEAR,
            start_test_year=TEST_START_YEAR,
            end_year=END_YEAR,
            num_shards=16,
        )


def build_data_module(batch_size=32):
    dm = cl.data.IterDataModule(
        task="downscaling",
        inp_root_dir=PROCESSED_LOW_DIR,
        out_root_dir=PROCESSED_HIGH_DIR,
        in_vars=[VARIABLE],
        out_vars=[VARIABLE],
        src="era5",
        subsample=SUBSAMPLE_HOURS,
        batch_size=batch_size,
        num_workers=1,
    )
    dm.setup()
    return dm


def build_model(dm):
    # architecture="resnet" (vs. model="resnet") wraps the backbone with a
    # bilinear interpolation to the target resolution before applying the
    # ResNet -- a bare ResNet has no upsampling and will shape-mismatch
    # against the high-res target.
    return cl.load_downscaling_module(data_module=dm, architecture="resnet")


def visualize(model, dm, num_samples=2, save_dir=None):
    # IterDataModule is iterable-only (no indexable .test_dataset / .time
    # attributes like the old map-style DataModule), so samples are pulled
    # from the front of the test dataloader rather than by timestamp.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    loader = dm.test_dataloader()
    examples = []
    for batch in loader:
        x, y, _, _ = batch
        for i in range(x.shape[0]):
            examples.append((x[i : i + 1], y[i : i + 1]))
            if len(examples) >= num_samples:
                break
        if len(examples) >= num_samples:
            break

    fig, axes = plt.subplots(len(examples), 4, figsize=(30, 3 * len(examples)), squeeze=False)

    for row, (x, y) in enumerate(examples):
        x, y = x.to(model.device), y.to(model.device)
        with torch.no_grad():
            pred = model.forward(x)

            denorm = model.test_target_transforms[0]
            low_res, high_res = denorm(x), denorm(y)
            pred = denorm(pred)
            bias = pred - high_res

        for col, tensor in enumerate([low_res, high_res, pred, bias]):
            ax = axes[row][col]
            im = ax.imshow(tensor.detach().squeeze().cpu().numpy())
            im.set_cmap(cmap=plt.cm.RdBu)
            fig.colorbar(im, ax=ax)

        axes[row][0].set_title("Low resolution data [Kelvin]")
        axes[row][1].set_title("High resolution data [Kelvin]")
        axes[row][2].set_title("Downscaled [Kelvin]")
        axes[row][3].set_title("Bias [Kelvin]")

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "visualize.png"))
    else:
        plt.show()


def visualize_mean_bias(model, dm, save_dir=None):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    loader = dm.test_dataloader()

    all_mean_bias = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x, y, _, _ = batch
            x = x.to(model.device)
            y = y.to(model.device)
            pred = model.forward(x)

            denorm = model.test_target_transforms[0]
            gt = denorm(y)
            pred = denorm(pred)
            bias = pred - gt
            all_mean_bias.append(bias.mean(dim=0))

    mean_bias = torch.stack(all_mean_bias, dim=0).mean(dim=0)

    fig, axes = plt.subplots(1, 1, figsize=(12, 4), squeeze=False)
    ax = axes[0, 0]
    im = ax.imshow(mean_bias.detach().squeeze().cpu().numpy())
    im.set_cmap(cmap=plt.cm.RdBu)
    fig.colorbar(im, ax=ax)
    ax.set_title("Mean bias [Kelvin]")
    fig.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "visualize_mean_bias.png"))
    else:
        plt.show()


def get_latest_checkpoint():
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.ckpt")), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


class PlainTextProgress(pl.callbacks.Callback):
    """Prints a new line every few batches -- the default rich/tqdm progress
    bar overwrites a single line with carriage returns, which doesn't survive
    being piped/redirected to a log file."""

    def __init__(self, print_every=5):
        self.print_every = print_every
        self._t_epoch = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._t_epoch = time.time()
        print(f"-- epoch {trainer.current_epoch} start --", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.print_every == 0:
            elapsed = time.time() - self._t_epoch
            loss = outputs["loss"].item() if isinstance(outputs, dict) else float(outputs)
            print(
                f"  epoch {trainer.current_epoch} batch {batch_idx} "
                f"loss={loss:.4f} elapsed={elapsed:.0f}s",
                flush=True,
            )

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"-- epoch {trainer.current_epoch} done in {time.time() - self._t_epoch:.0f}s --", flush=True)


def make_trainer(max_epochs=5):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="{epoch}",
        save_top_k=-1,
        every_n_epochs=1,
    )
    return pl.Trainer(
        accelerator="mps",
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, PlainTextProgress(print_every=5)],
        default_root_dir=PROJECT_DIR,
    )


def stage_prepare():
    if os.path.isdir(PROCESSED_LOW_DIR) and os.path.isdir(PROCESSED_HIGH_DIR):
        print("Already prepared (processed_low/processed_high exist) -- skipping.")
        return
    download()
    organize_into_variable_subfolders()
    convert()
    print("Done. Run `python downscaling.py train` next.")


def stage_train(resume=False, max_epochs=5, batch_size=16):
    dm = build_data_module(batch_size=batch_size)
    model = build_model(dm)
    trainer = make_trainer(max_epochs=max_epochs)
    ckpt_path = get_latest_checkpoint() if resume else None
    if resume and ckpt_path is None:
        print("No checkpoint found to resume from -- starting fresh.")
    trainer.fit(model, dm, ckpt_path=ckpt_path)
    print(f"Done. Checkpoints in {CHECKPOINT_DIR}. Run `evaluate` or `visualize` next.")


def stage_evaluate(batch_size=16):
    ckpt_path = get_latest_checkpoint()
    if ckpt_path is None:
        raise SystemExit("No checkpoint found -- run `train` first.")
    dm = build_data_module(batch_size=batch_size)
    model = build_model(dm)
    trainer = pl.Trainer(accelerator="mps")
    trainer.test(model, dm, ckpt_path=ckpt_path)


def stage_visualize(batch_size=16):
    ckpt_path = get_latest_checkpoint()
    if ckpt_path is None:
        raise SystemExit("No checkpoint found -- run `train` first.")
    dm = build_data_module(batch_size=batch_size)
    model = build_model(dm)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    visualize(model, dm, num_samples=2, save_dir=OUTPUTS_DIR)
    visualize_mean_bias(model, dm, save_dir=OUTPUTS_DIR)
    print(f"Saved plots to {OUTPUTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("prepare", help="download + convert the dataset (one-off)")

    train_p = sub.add_parser("train", help="train the model, checkpointing each epoch")
    train_p.add_argument("--resume", action="store_true", help="continue from latest checkpoint")
    train_p.add_argument("--epochs", type=int, default=5)
    train_p.add_argument("--batch-size", type=int, default=16)

    eval_p = sub.add_parser("evaluate", help="run trainer.test on the latest checkpoint")
    eval_p.add_argument("--batch-size", type=int, default=16)

    vis_p = sub.add_parser("visualize", help="produce outputs/*.png from the latest checkpoint")
    vis_p.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()

    if args.stage == "prepare":
        stage_prepare()
    elif args.stage == "train":
        stage_train(resume=args.resume, max_epochs=args.epochs, batch_size=args.batch_size)
    elif args.stage == "evaluate":
        stage_evaluate(batch_size=args.batch_size)
    elif args.stage == "visualize":
        stage_visualize(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
