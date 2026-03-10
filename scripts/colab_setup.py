"""
UGIF — Google Colab Training Script
=====================================
INSTRUCTIONS
  1. Open a new Colab notebook: https://colab.research.google.com
  2. Runtime → Change runtime type → T4 GPU
  3. Copy each CELL block below into its own Colab cell, then run top-to-bottom.
     (Or: File → Upload notebook and paste all cells at once.)

STORAGE NOTE:
  Everything lives on Colab's /content disk (~78 GB free).
  Google Drive is NOT required.

DATASET:
  TorchGeo will automatically download LEVIR-CD+ (~1.5 GB) on the first
  training run. It contains 985 pairs of 1024×1024 RGB bitemporal images
  (train/val/test splits), fully sufficient for high-quality training.

EXPECTED RUNTIME (T4 GPU):
  ~25 min per epoch × 50 epochs ≈ ~20 hours total.
  Use "Resume" cell to continue after a session timeout.
"""

# ════════════════════════════════════════════════════════════════
# CELL 1 ─ Check GPU + disk space
# ════════════════════════════════════════════════════════════════
import subprocess, sys, os, shutil

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout or "No GPU found — switch runtime to GPU!")

# Show available disk space
subprocess.run(["df", "-h", "/content"], check=True)


# ════════════════════════════════════════════════════════════════
# CELL 2 ─ Clone UGIF repo
# ════════════════════════════════════════════════════════════════
REPO_URL    = "https://github.com/rennnss/UGIF.git"
PROJECT_DIR = "/content/ugif"
DATA_DIR    = "/content/data"       # LEVIR-CD+ lives here (~1.5 GB)
OUTPUT_DIR  = "/content/outputs"   # checkpoints + TensorBoard logs

os.chdir("/content")

# Fresh clone every time to pick up latest code
if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)
os.system(f"git clone --depth 1 {REPO_URL} {PROJECT_DIR}")

# Optional: copernicus_api for real SAR downloads
os.makedirs(f"{PROJECT_DIR}/third_party", exist_ok=True)
os.system(
    "git clone --depth 1 "
    "https://github.com/armkhudinyan/copernicus_api.git "
    f"{PROJECT_DIR}/third_party/copernicus_api"
)

# Create output directories
for d in [DATA_DIR, f"{OUTPUT_DIR}/checkpoints", f"{OUTPUT_DIR}/logs"]:
    os.makedirs(d, exist_ok=True)

print("\n✅ Repo cloned")
print("Contents:", os.listdir(PROJECT_DIR))


# ════════════════════════════════════════════════════════════════
# CELL 3 ─ Install Python dependencies
#           Colab ships with PyTorch + CUDA already — skip those.
# ════════════════════════════════════════════════════════════════
packages = [
    "pytorch-lightning>=2.0",
    "hydra-core", "omegaconf",
    "torchgeo>=0.5",           # LEVIR-CD+ dataset + data utilities
    "shap",
    "geopandas", "geopy",
    "rasterio",
    "python-dotenv",
    "spacy",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + packages,
    check=True,
)

# Language model for the NL query interface
subprocess.run(
    [sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"],
    check=True,
)

# Install the UGIF project in editable mode so `src.*` is importable
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", PROJECT_DIR],
    check=True,
)

import torch
print(f"\nPyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ════════════════════════════════════════════════════════════════
# CELL 4 ─ Verify imports (fast sanity check)
# ════════════════════════════════════════════════════════════════
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Core project modules
from src.data.datamodule    import UGIFDataModule         # type: ignore
from src.training.lightning_module import UGIFLightningModule  # type: ignore
from src.models.siamese     import SiameseFCN             # type: ignore

# TorchGeo dataset (the actual data source)
from torchgeo.datasets import LEVIRCDPlus              # type: ignore

print("✅ All imports OK")


# ════════════════════════════════════════════════════════════════
# CELL 5 ─ Download LEVIR-CD+ dataset (~1.5 GB total)
#   TorchGeo downloads each split automatically. A retry loop
#   handles Colab's occasional network drops and bad zips.
# ════════════════════════════════════════════════════════════════
import urllib.error, zipfile, time, glob

def _download_split_with_retry(split: str, root: str, max_retries: int = 6) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            LEVIRCDPlus(root=root, split=split, download=True)
            print(f"  [{split}] done ✅")
            return
        except (urllib.error.ContentTooShortError, urllib.error.URLError,
                OSError, zipfile.BadZipFile) as exc:
            print(f"  [{split}] attempt {attempt}/{max_retries} failed: {type(exc).__name__}")
            if attempt == max_retries:
                raise
            # Purge corrupt/partial files so TorchGeo re-downloads cleanly
            for f in glob.glob(os.path.join(root, "**", "*.zip"), recursive=True):
                os.remove(f)
            extracted = os.path.join(root, "LEVIRCDPlus")
            if os.path.exists(extracted):
                shutil.rmtree(extracted)
            time.sleep(10 * attempt)

print("Downloading LEVIR-CD+ …")
for split in ("train", "test"):   # LEVIRCDPlus has no val split; val is carved from train
    _download_split_with_retry(split, root=DATA_DIR)

subprocess.run(["df", "-h", "/content"])


# ════════════════════════════════════════════════════════════════
# CELL 6 ─ Smoke test  (~30 seconds, 2 mini-batches, no real data needed)
#   Proves the full pipeline wires up before committing to 50 epochs.
# ════════════════════════════════════════════════════════════════
import pytorch_lightning as pl
from src.training.callbacks import get_callbacks           # type: ignore

pl.seed_everything(42, workers=True)

dm_smoke = UGIFDataModule(
    root=DATA_DIR,
    patch_size=256,
    batch_size=4,
    num_workers=2,
)
model_smoke = UGIFLightningModule(
    in_channels=5,    # 3 RGB optical + 2 synthetic SAR
    num_features=4,
    max_epochs=1,
)
trainer_smoke = pl.Trainer(
    max_epochs=1,
    limit_train_batches=2,
    limit_val_batches=2,
    accelerator="auto",
    devices="auto",
    enable_progress_bar=True,
    logger=False,
    enable_checkpointing=False,
)
trainer_smoke.fit(model_smoke, datamodule=dm_smoke)
print("\n✅ Smoke test passed — pipeline is wired up correctly")

del dm_smoke, model_smoke, trainer_smoke   # free memory


# ════════════════════════════════════════════════════════════════
# CELL 7 ─ Full training  (50 epochs, Optical + synthetic SAR)
#
#   Batch-size guide for free T4 (15 GB VRAM):
#     patch_size=256 → batch_size=8   (safe, recommended)
#     patch_size=256 → batch_size=16  (may OOM on larger backbones)
#
#   Early stopping is OFF (patience=0) so all 50 epochs always run.
#   Best checkpoint is saved; 'last.ckpt' is kept for resuming.
# ════════════════════════════════════════════════════════════════
MAX_EPOCHS  = 50
BATCH_SIZE  = 8
PATCH_SIZE  = 256
NUM_WORKERS = 2
IN_CHANNELS = 5    # 3 optical RGB + 2 synthetic SAR (VV / VH proxy)

pl.seed_everything(42, workers=True)

dm = UGIFDataModule(
    root=DATA_DIR,
    patch_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
model = UGIFLightningModule(
    in_channels=IN_CHANNELS,
    num_features=4,
    feature_dim=128,
    r_max=10.0,
    epsilon=1e-6,
    lr=1e-4,
    weight_decay=1e-5,
    margin=1.0,
    max_epochs=MAX_EPOCHS,
    lambda_csn=0.3,
)
callbacks = get_callbacks(
    output_dir=OUTPUT_DIR,
    patience=0,           # early stopping DISABLED — run all epochs
)
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    callbacks=callbacks,
    logger=pl.loggers.TensorBoardLogger(
        save_dir=f"{OUTPUT_DIR}/logs",
        name="ugif",
    ),
    accelerator="auto",
    devices="auto",
    precision="16-mixed",   # AMP halves VRAM use and speeds up T4
    deterministic=True,    # deterministic=True is slow on GPU; off for speed
    enable_progress_bar=True,
    log_every_n_steps=10,
)

print(f"\n🚀 Starting training: {MAX_EPOCHS} epochs | batch={BATCH_SIZE} | patch={PATCH_SIZE}")
trainer.fit(model, datamodule=dm)

print("\n✅ Training complete — running test evaluation on best checkpoint…")
trainer.test(model, datamodule=dm, ckpt_path="best")


# ════════════════════════════════════════════════════════════════
# CELL 8 ─ Resume after a Colab session timeout
#   Run ONLY this cell (after re-running CELL 1-4 to restore the
#   environment) to pick up from the last saved checkpoint.
# ════════════════════════════════════════════════════════════════
# LAST_CKPT = f"{OUTPUT_DIR}/checkpoints/last.ckpt"
#
# pl.seed_everything(42, workers=True)
# dm = UGIFDataModule(root=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
# model = UGIFLightningModule.load_from_checkpoint(LAST_CKPT)
# callbacks = get_callbacks(output_dir=OUTPUT_DIR, patience=0)
# trainer = pl.Trainer(
#     max_epochs=MAX_EPOCHS,
#     callbacks=callbacks,
#     logger=pl.loggers.TensorBoardLogger(save_dir=f"{OUTPUT_DIR}/logs", name="ugif"),
#     accelerator="auto", devices="auto", precision="16-mixed",
# )
# trainer.fit(model, datamodule=dm, ckpt_path=LAST_CKPT)
# trainer.test(model, datamodule=dm, ckpt_path="best")


# ════════════════════════════════════════════════════════════════
# CELL 9 ─ TensorBoard (inline)
# ════════════════════════════════════════════════════════════════
# %load_ext tensorboard
# %tensorboard --logdir /content/outputs/logs


# ════════════════════════════════════════════════════════════════
# CELL 10 ─ Show training metrics summary
# ════════════════════════════════════════════════════════════════
import glob
import csv

log_files = sorted(glob.glob(f"{OUTPUT_DIR}/logs/ugif/version_*/metrics.csv"))
if log_files:
    with open(log_files[-1]) as f:
        rows = list(csv.DictReader(f))
    # Print final epoch metrics
    val_rows = [r for r in rows if r.get("val_iou")]
    if val_rows:
        last = val_rows[-1]
        print(f"\n{'─'*40}")
        print(f"  Final val IoU  : {float(last['val_iou']):.4f}")
        print(f"  Final val Loss : {float(last.get('val_loss', 0)):.4f}")
        print(f"{'─'*40}")
else:
    print("No metrics.csv found yet — training may not have completed.")


# ════════════════════════════════════════════════════════════════
# CELL 11 ─ NL query + DII explainability report (optional)
# ════════════════════════════════════════════════════════════════
os.system(
    f"cd {PROJECT_DIR} && python src/frontend/llm_agent.py "
    '--query "flood damage in Chennai August 2023" --geojson'
)
os.system(f"cd {PROJECT_DIR} && python src/explainability/report_generator.py")
