"""
UGIF — Google Colab Setup Script (FIXED)
========================================
Run each numbered cell in order in a Colab notebook.
Runtime: Runtime → Change runtime type → T4 GPU (free)

Repo: https://github.com/rennnss/UGIF
"""

# ──────────────────────────────────────────────────────────────
# CELL 1 — Configuration (edit these)
# ──────────────────────────────────────────────────────────────
REPO_URL    = "https://github.com/rennnss/UGIF.git"
PROJECT_DIR = "/content/ugif"
DRIVE_ROOT  = "/content/drive/MyDrive/UGIF"


# ──────────────────────────────────────────────────────────────
# CELL 2 — Mount Google Drive
# ──────────────────────────────────────────────────────────────
from google.colab import drive
import os

drive.mount('/content/drive')
os.makedirs(DRIVE_ROOT, exist_ok=True)
os.makedirs(f"{DRIVE_ROOT}/data", exist_ok=True)
os.makedirs(f"{DRIVE_ROOT}/outputs", exist_ok=True)
print("Drive mounted.")


# ──────────────────────────────────────────────────────────────
# CELL 3 — Clone the UGIF project + copernicus_api dependency
# ──────────────────────────────────────────────────────────────
import os, shutil

# Reset cwd — Colab shell can get stuck in a deleted directory
os.chdir('/content')

# Clean up any partial clone from a previous failed attempt
if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)

os.system(f"git clone {REPO_URL} {PROJECT_DIR}")
os.makedirs(f"{PROJECT_DIR}/third_party", exist_ok=True)
os.system(f"git clone https://github.com/armkhudinyan/copernicus_api.git "
          f"{PROJECT_DIR}/third_party/copernicus_api")

print(f"\nProject ready at {PROJECT_DIR}")
print(os.listdir(PROJECT_DIR))


# ──────────────────────────────────────────────────────────────
# CELL 4 — Install dependencies
#           (Colab already has torch + CUDA — skip those)
# ──────────────────────────────────────────────────────────────
!pip install -q \
    pytorch-lightning \
    hydra-core omegaconf \
    shap \
    geopandas geopy rasterio \
    python-dotenv \
    spacy

!python -m spacy download en_core_web_sm -q

# Install the UGIF project itself so 'src' is importable everywhere
!pip install -q -e {PROJECT_DIR}

import torch
print(f"\nPyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ──────────────────────────────────────────────────────────────
# CELL 5 — Verify src package imports correctly
# ──────────────────────────────────────────────────────────────
import sys
os.chdir(PROJECT_DIR)

# Now 'src' is a proper installed package — no manual sys.path hacks needed
from src.data.sar_downloader import SARDownloader     # type: ignore
from src.data.levir_dataset import LEVIRCDPatchDataset # type: ignore
from src.models.siamese import SiameseFCN              # type: ignore
from src.explainability.dii import compute_dii_improved# type: ignore

print("✅ All UGIF imports OK")


# ──────────────────────────────────────────────────────────────
# CELL 6 — Set Copernicus credentials
#           Use Colab's built-in Secrets (left sidebar → 🔑)
#           Add: COPERNICUS_USER and COPERNICUS_PASS
# ──────────────────────────────────────────────────────────────
from google.colab import userdata

os.environ['COPERNICUS_USER'] = userdata.get('COPERNICUS_USER')
os.environ['COPERNICUS_PASS'] = userdata.get('COPERNICUS_PASS')
print("Credentials loaded from Colab Secrets ✅")


# ──────────────────────────────────────────────────────────────
# CELL 7 — Download Sentinel-1 SAR data
#           Data saved to Drive so it persists after session ends.
# ──────────────────────────────────────────────────────────────
SAR_OUT = f"{DRIVE_ROOT}/data/SAR/area_2023"
os.makedirs(SAR_OUT, exist_ok=True)

dl = SARDownloader()  # picks up env vars set above
dl.download(
    bbox=(80.0, 12.0, 81.0, 13.5),   # ← change to your area of interest
    start_date="2023-08-01",
    end_date="2023-08-31",
    out_dir=SAR_OUT,
    prod_type="GRD",
    orbit_direction="ASCENDING",
    max_products=5,
)


# ──────────────────────────────────────────────────────────────
# CELL 8 — Symlink Drive directories into project
#           Checkpoints + data persist between Colab sessions.
# ──────────────────────────────────────────────────────────────
def safe_symlink(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)
        print(f"Linked {dst} → {src}")
    else:
        print(f"Already exists: {dst}")

safe_symlink(f"{DRIVE_ROOT}/outputs", f"{PROJECT_DIR}/outputs")
safe_symlink(f"{DRIVE_ROOT}/data",    f"{PROJECT_DIR}/data")


# ──────────────────────────────────────────────────────────────
# CELL 9 — Smoke test (2 steps, ~30 sec, no real data needed)
# ──────────────────────────────────────────────────────────────
os.chdir(PROJECT_DIR)
!python src/training/train.py training.fast_dev_run=True


# ──────────────────────────────────────────────────────────────
# CELL 10 — Full training
#
#   GPU presets (data.num_workers=2 keeps Colab stable):
#     T4  (free):  batch_size=16, patch_size=256
#     L4  (Pro):   batch_size=24, patch_size=256
#     A100(Pro+):  batch_size=32, patch_size=512
# ──────────────────────────────────────────────────────────────
os.chdir(PROJECT_DIR)
!python src/training/train.py \
    data.root=./data/LEVIR-CD \
    data.batch_size=16 \
    data.patch_size=256 \
    data.num_workers=2 \
    training.max_epochs=50 \
    output.dir=./outputs \
    output.log_dir=./outputs/logs

# To RESUME after a session timeout (checkpoint is on Drive):
# !python src/training/train.py ckpt_path=./outputs/checkpoints/last.ckpt


# ──────────────────────────────────────────────────────────────
# CELL 11 — TensorBoard (inline in Colab)
# ──────────────────────────────────────────────────────────────
# %load_ext tensorboard
# %tensorboard --logdir {PROJECT_DIR}/outputs/logs


# ──────────────────────────────────────────────────────────────
# CELL 12 — NL query + DII report
# ──────────────────────────────────────────────────────────────
os.chdir(PROJECT_DIR)
!python src/frontend/llm_agent.py \
    --query "flood damage in Chennai August 2023" \
    --geojson

!python src/explainability/report_generator.py
