"""
UGIF — Google Colab Setup Script (Local Storage Only)
======================================================
All data, checkpoints and logs are kept on Colab's ephemeral /content disk.
Nothing is written to Google Drive — no mounting required.

Run each numbered cell in order in a Colab notebook.
Runtime: Runtime → Change runtime type → T4 GPU (free)

Repo: https://github.com/rennnss/UGIF
"""

# ──────────────────────────────────────────────────────────────
# CELL 1 — Configuration
# ──────────────────────────────────────────────────────────────
REPO_URL    = "https://github.com/rennnss/UGIF.git"
PROJECT_DIR = "/content/ugif"
DATA_DIR    = f"{PROJECT_DIR}/data"         # LEVIR-CD downloaded here by TorchGeo
OUTPUT_DIR  = f"{PROJECT_DIR}/outputs"      # checkpoints + logs (local only)


# ──────────────────────────────────────────────────────────────
# CELL 2 — Clone the UGIF project
# ──────────────────────────────────────────────────────────────
import os, shutil

# Reset cwd — Colab shell can get stuck in a deleted directory
os.chdir('/content')

# Clean up any partial clone from a previous failed attempt
if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)

os.system(f"git clone {REPO_URL} {PROJECT_DIR}")
os.makedirs(f"{PROJECT_DIR}/third_party", exist_ok=True)
os.system(
    f"git clone https://github.com/armkhudinyan/copernicus_api.git "
    f"{PROJECT_DIR}/third_party/copernicus_api"
)

# Create local output directories
os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"\nProject ready at {PROJECT_DIR}")
print(os.listdir(PROJECT_DIR))


# ──────────────────────────────────────────────────────────────
# CELL 3 — Install dependencies
#           (Colab already has torch + CUDA — skip those)
# ──────────────────────────────────────────────────────────────
import subprocess, sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "pytorch-lightning",
    "hydra-core", "omegaconf",
    "shap",
    "geopandas", "geopy", "rasterio",
    "python-dotenv",
    "spacy",
    "torchgeo",
], check=True)

subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=True)

# Install the UGIF project itself so 'src' is importable everywhere
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", PROJECT_DIR], check=True)

import torch
print(f"\nPyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ──────────────────────────────────────────────────────────────
# CELL 4 — Verify imports
# ──────────────────────────────────────────────────────────────
import sys
os.chdir(PROJECT_DIR)

from src.data.sar_downloader import SARDownloader       # type: ignore
from torchgeo.datasets import LEVIRCDPlus               # type: ignore
from src.models.siamese import SiameseFCN               # type: ignore
from src.explainability.dii import compute_dii_improved # type: ignore

print("✅ All UGIF imports OK")


# ──────────────────────────────────────────────────────────────
# CELL 5 — (Optional) Set Copernicus credentials for SAR download
#           Use Colab's built-in Secrets (left sidebar → 🔑)
#           Add: COPERNICUS_USER and COPERNICUS_PASS
#           Skip this cell if you are only using LEVIR-CD (no real SAR needed).
# ──────────────────────────────────────────────────────────────
try:
    from google.colab import userdata
    os.environ['COPERNICUS_USER'] = userdata.get('COPERNICUS_USER')
    os.environ['COPERNICUS_PASS'] = userdata.get('COPERNICUS_PASS')
    print("Credentials loaded from Colab Secrets ✅")
except Exception as e:
    print(f"Skipping Copernicus credentials: {e}")


# ──────────────────────────────────────────────────────────────
# CELL 6 — Smoke test (1 batch, ~30 sec — confirms everything wires up)
# ──────────────────────────────────────────────────────────────
os.system(
    f"cd {PROJECT_DIR} && python src/training/train.py "
    f"training.fast_dev_run=True "
    f"data.root={DATA_DIR} "
    f"output.dir={OUTPUT_DIR} "
    f"output.log_dir={OUTPUT_DIR}/logs"
)


# ──────────────────────────────────────────────────────────────
# CELL 7 — Full training (50 epochs)
#
#   NOTE: TorchGeo will automatically download LEVIR-CD+ (~700 MB)
#         into DATA_DIR on first run.
#
#   GPU presets (data.num_workers=2 keeps Colab stable):
#     T4  (free):  batch_size=8,  patch_size=256
#     L4  (Pro):   batch_size=16, patch_size=256
#     A100(Pro+):  batch_size=32, patch_size=512
# ──────────────────────────────────────────────────────────────
os.system(
    f"cd {PROJECT_DIR} && python src/training/train.py "
    f"data.root={DATA_DIR} "
    f"data.batch_size=8 "
    f"data.patch_size=256 "
    f"data.num_workers=2 "
    f"training.max_epochs=50 "
    f"training.patience=0 "
    f"output.dir={OUTPUT_DIR} "
    f"output.log_dir={OUTPUT_DIR}/logs"
)

# ── RESUME after a session timeout (ckpt saved to local disk) ──
# os.system(
#     f"cd {PROJECT_DIR} && python src/training/train.py "
#     f"data.root={DATA_DIR} "
#     f"data.num_workers=2 "
#     f"training.max_epochs=50 "
#     f"ckpt_path={OUTPUT_DIR}/checkpoints/last.ckpt "
#     f"output.dir={OUTPUT_DIR} "
#     f"output.log_dir={OUTPUT_DIR}/logs"
# )


# ──────────────────────────────────────────────────────────────
# CELL 8 — TensorBoard (inline in Colab)
# ──────────────────────────────────────────────────────────────
# %load_ext tensorboard
# %tensorboard --logdir /content/ugif/outputs/logs


# ──────────────────────────────────────────────────────────────
# CELL 9 — NL query + DII explainability report
# ──────────────────────────────────────────────────────────────
os.system(
    f"cd {PROJECT_DIR} && python src/frontend/llm_agent.py "
    f"--query \"flood damage in Chennai August 2023\" --geojson"
)
os.system(f"cd {PROJECT_DIR} && python src/explainability/report_generator.py")
