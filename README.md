# BacteriaCV

BacteriaCV is a computer vision project for classifying bacterial cell shape and Gram status from microscope images. The repository currently focuses on preparing the DIBaS dataset: downloading the archives, extracting images, mapping species, and building a leak-free `data/index.csv`.

## Project status

- Full extraction yields 692 images from 33 species labels.
- `Candida albicans` is excluded from training because it is a fungus, not a bacterium, leaving 32 trainable species.
- `data/index.csv` holds 669 images: 467 train, 136 validation, and 66 test.
- The five folds are built from train data only.
- Species names are mapped through `bacteriacv/datasets/species_map.py` rather than guessed from filenames.
- The training module and the web interface are not yet part of this dataset stage.

## Main structure

```text
BacteriaCV/
├── bacteriacv/
│   ├── config.py
│   ├── label_map.py
│   ├── paths.py
│   └── datasets/
│       ├── species_map.py
│       ├── download.py
│       ├── extract.py
│       └── build_index.py
├── data/
│   ├── index.csv
│   └── raw/
│       └── zips_manifest.csv
├── docs/
├── scripts/
└── tests/
```

## Environment setup

In PowerShell from the repository root:

```powershell
.\scripts\setup_env.ps1
.\scripts\check_env.ps1
```

Or use an existing Python environment and install the project dependencies as needed.

## Installation

Requires Python 3.11.0. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The CPU index for torch is already handled inside `requirements.txt` through `--extra-index-url`, so there is no extra command. `torch` and `torchvision` use the local version suffix `+cpu` which does not exist on PyPI, and without that extra index `pip install` fails with `No matching distribution found`.

The application also needs a `checkpoints/` folder containing `heads.pt` and `evaluation.json`. Without `heads.pt` the application stops at startup with `FileNotFoundError`. Without `evaluation.json`, `/api/report` returns 404 "Evaluation has not been run yet".

## Running the application

```sh
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

`${PORT:-8000}` is deliberate. Railway injects the `PORT` variable dynamically and then routes traffic to that number, and the `:-8000` form falls back to 8000 when `PORT` is absent, so the same command works on the server and on a local machine. Therefore never write a bare `--port 8000` anywhere; a hardcoded port means Railway cannot reach the application. The `${PORT:-8000}` form is POSIX shell expansion. In PowerShell, set the port first: `$env:PORT = "8123"`.

Check health:

```
http://127.0.0.1:8000/api/health
```

The expected response once the model is loaded:

```json
{"status":"ready","checkpoint":"heads.pt","max_upload_bytes":20971520}
```

Upload limit is 20 MB. The application accepts files larger than 5 MB, so never write 5 MB anywhere. Accepted formats: `png`, `jpg`, `jpeg`, `tif`, `tiff`.

The `heads.pt` file in the repository root is not the model in use and has been deleted. Its weights differ from `checkpoints/heads.pt` and were never read by the application. What gets loaded is always `checkpoints/heads.pt`.

## Dataset pipeline

Download the 33 DIBaS archives and record the SHA-256 manifest:

```powershell
python -m bacteriacv.datasets.download
```

Extract the images and verify the result:

```powershell
python -m bacteriacv.datasets.extract
python -m bacteriacv.datasets.extract --verify-only
```

Build the dataset index:

```powershell
python -m bacteriacv.datasets.build_index
```

`data/index.csv` contains relative paths, species names, `species_id`, split, and fold. Validation and test data use `fold=-1`.

Note: the original DIBaS server has an expired TLS certificate. The download module uses a connection without certificate verification for that source and records the SHA-256 of every archive in `data/raw/zips_manifest.csv` as provenance evidence.

## Testing

```powershell
python -m pytest -q
```

Tests cover species mapping, data locations, downloading, extraction, dataset splitting, leakage checks, folds, and relative paths.

## Documentation

- [SPEC](docs/SPEC.md) — source of truth for project requirements.
- [ARCHITECTURE](docs/ARCHITECTURE.md) — architecture and technical decisions.
- [DESIGN](docs/DESIGN.md) — interface design and module contracts.
- [AGENT](AGENT.md) — contribution and audit rules.

## Running the online version

The full version of the application is running at:

```text
https://bacteriacv-demo.up.railway.app
```

Upload images in png, jpg, jpeg, tif, or tiff format with a 20 MB limit. A health check is available at `/api/health` and the model evaluation report at `/api/report`.

To try it without preparing your own images, the `docs/contoh/` folder holds four tested PNG images at 768 x 574 that can be uploaded directly to the upload field.

The sample images are derived from the DIBaS dataset (Zielinski et al. 2017, PLOS ONE 12(9):e0184554), reduced to 768 x 574 for demonstration purposes. The Gram status the platform reports for a sample image may differ from the result on the original image because of the scale difference.

## Additional image data

Besides the four sample images in `docs/contoh/`, a more complete set of images and archives is available on Google Drive:

```text
https://drive.google.com/drive/folders/1dSyJUjPfyzu5GrCnH1LWZ_qRRSmnviA_?usp=drive_link
```

That folder contains images classified per species as well as the original ZIP archives used to build the dataset pipeline. The formats the application accepts are png, jpg, jpeg, tif, and tiff with a 20 MB limit per file.

To rebuild the pipeline from the ZIP archives:

```bash
python -m bacteriacv.datasets.download
python -m bacteriacv.datasets.extract
python -m bacteriacv.datasets.build_index
```

Those commands download the archives from their original source, not from Google Drive. Use the Drive folder when you already have the images or archives.

## License

This project is released under the [MIT License](LICENSE).