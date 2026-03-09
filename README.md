# UGIF — Unified Geospatial Intelligence Framework

Disaster damage assessment pipeline using a Fully Convolutional Siamese Network (FC-CSN) with SHAP-based Disaster Impact Index (DII).

## Architecture

```
Input (Google Earth + SAR)
        ↓
   Preprocessing
        ↓
  CNN + CSN (Siamese FCN)
   ├── CNN OUTPUT: Class label / feature map  (e.g. "Urban — 0.82")
   └── CSN OUTPUT: Similarity score           (e.g. distance = 0.15)
        ↓
DII_improved = (1/N) Σ_g Σ_i φᵢⁿᵒʳᵐ · |f̄(O_pre, S_pre)_{g,i} + ε|
                                          ─────────────────────────────
                                          |f̄(O_post, S_post)_{g,i} + ε|
        ↓
Before/After damage visualisation
```

## Quick Start (with synthetic data — no download needed)

```bash
# 1. Create virtual environment
cd /Users/ren/minor_project
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Run tests (verifies full pipeline)
python -m pytest tests/ -v

# 4. Smoke-test training (synthetic data, 2 steps)
python src/training/train.py training.fast_dev_run=True
```

## Real Data Setup

### Google Earth Optical Imagery (LEVIR-CD)

Download from the [LEVIR-CD official page](https://justchenhao.github.io/LEVIR/) and place as:

```
data/LEVIR-CD/
  train/{A,B,label}/   # pre, post, mask .png files
  val/{A,B,label}/
  test/{A,B,label}/
```

### SAR Data — Sentinel-1 via copernicus_api (free, CDSE)

The project uses **[armkhudinyan/copernicus_api](https://github.com/armkhudinyan/copernicus_api)**,
already cloned into `third_party/copernicus_api`. It queries the
**Copernicus Data Space Ecosystem (CDSE)** — free account at https://dataspace.copernicus.eu/

**1. Create a `.env` file in the project root:**
```bash
COPERNICUS_USER=your@email.com
COPERNICUS_PASS=yourpassword
```

**2. Download via the CLI script (bbox):**
```bash
python scripts/download_sar.py \
    --bbox 80.0 12.0 81.0 13.5 \
    --start 2023-08-01 --end 2023-08-31 \
    --out data/SAR/chennai_2023 \
    --orbit ASCENDING \
    --max 10
```

**3. Or use a GeoJSON footprint file:**
```bash
python scripts/download_sar.py \
    --geojson path/to/area.geojson \
    --start 2023-08-01 --end 2023-08-31
```

**4. Or use the Python API directly:**
```python
from src.data.sar_downloader import SARDownloader

dl = SARDownloader()   # reads COPERNICUS_USER/PASS from .env
dl.download(
    bbox=(80.0, 12.0, 81.0, 13.5),
    start_date="2023-08-01",
    end_date="2023-08-31",
    out_dir="data/SAR/chennai_2023",
    orbit_direction="ASCENDING",
    max_products=10,
)
```

Downloaded `.zip` files (Sentinel-1 SAFE format) go into `data/SAR/<location>/`.

## Training

```bash
# Full training with real LEVIR-CD data
python src/training/train.py \
  data.root=./data/LEVIR-CD \
  data.batch_size=8 \
  training.max_epochs=50

# Monitor with TensorBoard
tensorboard --logdir outputs/logs/ugif
```

## Inference & DII Report

```bash
# Natural language query → GeoJSON
python src/frontend/llm_agent.py \
  --query "flood damage in Chennai August 2023" \
  --geojson

# Generate damage assessment report
python src/explainability/report_generator.py
```

Optional LLM upgrade: set `OPENAI_API_KEY` to activate GPT-4o-mini query parsing.

## Project Structure

```
minor_project/
├── configs/default.yaml          # Hydra config
├── src/
│   ├── data/
│   │   ├── levir_dataset.py      # LEVIR-CD wrapper (synthetic fallback)
│   │   ├── fusion.py             # X_fused = O_i ⊕ S_i
│   │   ├── transforms.py         # Augmentation + normalisation
│   │   └── datamodule.py         # LightningDataModule
│   ├── models/
│   │   ├── fcn.py                # FCN encoder → f̄(O,S)_{g,i}
│   │   ├── siamese.py            # Siamese FCN + ratio R_{g,i}
│   │   └── losses.py             # CSN + Dice + BCE losses
│   ├── training/
│   │   ├── lightning_module.py   # UGIFLightningModule
│   │   ├── callbacks.py          # Checkpoint, EarlyStopping, LR monitor
│   │   └── train.py              # Hydra entrypoint
│   ├── explainability/
│   │   ├── dii.py                # DII_improved formula
│   │   ├── shap_explainer.py     # SHAP φᵢⁿᵒʳᵐ computation
│   │   └── report_generator.py   # Markdown report
│   ├── frontend/
│   │   ├── query_parser.py       # NL → structured dict
│   │   ├── geojson_builder.py    # Nominatim geocoding → GeoJSON
│   │   └── llm_agent.py          # CLI agent
│   └── utils/
│       ├── metrics.py            # IoU, F1, precision/recall
│       └── visualization.py      # Damage maps + SHAP charts
└── tests/                        # 28 unit tests
```
# UGIF
