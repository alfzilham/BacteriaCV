<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mark-dark.svg">
  <img src="apps/site/public/icons/mark.svg" alt="BacteriaCV logo" width="96" height="99">
</picture>

# BacteriaCV

**Cell shape and Gram status from one microscope image.**
Open source, trained on public data.

[![CI](https://github.com/alfzilham/BacteriaCV/actions/workflows/ci.yml/badge.svg)](https://github.com/alfzilham/BacteriaCV/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=nextdotjs)
![FastAPI](https://img.shields.io/badge/FastAPI-PyTorch-009688?logo=fastapi&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-Android-02569B?logo=flutter&logoColor=white)

[Web app](https://bacteriacv.vercel.app/app) |
[Documentation](https://bacteriacv.vercel.app/docs/quick-start) |
[API reference](https://bacteriacv.vercel.app/docs/api) |
[Changelog](https://bacteriacv.vercel.app/docs/changelog)

</div>

> [!IMPORTANT]
> This result is an aid, not a diagnosis. Segmentation groups cells, it does not isolate individual cells.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [How it works](#how-it-works)
- [Model and data](#model-and-data)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Self-hosting](#self-hosting)
- [API reference](#api-reference)
- [Testing and CI](#testing-and-ci)
- [Releases and changelog](#releases-and-changelog)
- [Project status](#project-status)
- [Contributing](#contributing)
- [License](#license)

## Overview

BacteriaCV takes one microscope image of stained bacteria and returns two answers: the cell shape (cocci or bacilli) and the Gram status (positive or negative). Each answer comes with a confidence level, and the image is shown through five processing stages so you can see how it was read.

The project is a monorepo with three apps that share one model:

- a **website** with a landing page, documentation, a changelog, and the web app at `/app`;
- an **inference API** built with FastAPI and PyTorch;
- an **Android app** built with Flutter that runs the model on the device.

## Features

- **Two predictions per image.** Cell shape and Gram status, each with a confidence level (High, Medium, or Low).
- **Five processing stages.** Original, resized, normalized, segmented, and watershed views, so every result can be inspected.
- **Honest by design.** Every result carries a fixed disclaimer, and low-confidence results ask for a standard stain to confirm.
- **Bilingual interface.** English and Indonesian, with light, dark, and system themes.
- **Installable web app.** The site ships a web app manifest and a service worker.
- **Self-hostable.** Clone the repository, run the API locally, or deploy your own copy.
- **Changelog from GitHub Releases.** The website reads release notes directly from this repository.

## How it works

1. You choose a PNG, JPG, or TIFF image of up to 20 MB.
2. The API preprocesses it and runs a frozen ResNet-50 feature extractor.
3. Two small heads read the features: one for cell shape, one for Gram status.
4. The API returns both predictions, five stage panels, and notes about any stage that failed.

| # | Stage | What you see |
|---|-------|--------------|
| 1 | `original` | The uploaded image |
| 2 | `resized` | The image resized for the model |
| 3 | `normalized` | The image after normalization |
| 4 | `segment` | The segmentation result |
| 5 | `watershed` | The watershed split of touching cells |

Confidence levels:

| Level | Confidence | Meaning |
|-------|------------|---------|
| High | 0.8 or above | The model is sure |
| Medium | 0.6 to 0.79 | Treat with care |
| Low | below 0.6 | Check with a standard stain |

## Model and data

| Item | Detail |
|------|--------|
| Backbone | ResNet-50 with ImageNet V2 weights, frozen |
| Heads | Two `Linear(2048, 2)` layers: shape (cocci, bacilli) and Gram (positive, negative) |
| Dataset | DIBaS, 32 trainable species |
| Images | 692 extracted from 33 species labels; 669 in `data/index.csv` (467 train, 136 validation, 66 test) |
| Excluded | `Candida albicans`, because it is a fungus, not a bacterium |
| Checkpoint | `apps/api/checkpoints/heads.pt` (heads only, the backbone is downloaded by torchvision) |
| Checkpoint SHA-256 | `46f194b368603c3ad2ac65f5aff7e7dfbf3476104bbac41944060aa3c6bbbd76` |

On the held-out test split (66 images, 32 species) the released checkpoint reaches macro F1 of 1.00 and accuracy of 1.00 for both heads. The full report is in `apps/api/checkpoints/evaluation.json`.

Please read these numbers with care. The test split is small and comes from the same public dataset as the training images, so it says little about images from other microscopes, stains, or laboratories. DIBaS has no spiral-shaped species, so a spiral class is not predicted and is excluded from the metrics.

Dataset preparation (download, extraction, species mapping, and the leak-free index) is described in [`apps/api/README.md`](apps/api/README.md).

## Repository layout

| Folder | Contents | Deploy target |
|--------|----------|---------------|
| `apps/site` | Landing page, documentation, changelog, and web app (Next.js 15, React 19, TypeScript) | Vercel |
| `apps/api` | Inference API (FastAPI and PyTorch). Model package in `bacteriacv/`, controllers in `app/controllers`, services in `app/services` | Railway |
| `apps/mobile` | Android app (Flutter). Code in `lib/models`, `lib/views`, `lib/controllers`, and `lib/theme` | APK |
| `docs/` | Review notes and archived v1 specifications | none |
| `tools/` | Repository policy check and publishing helper | none |
| `.github/workflows` | Continuous integration | none |

Each app follows the same idea: models hold data and logic, views draw the interface, and controllers connect the two.

## Getting started

### Prerequisites

- Python 3.11 to 3.13 for the API
- Node.js 20 or newer for the site
- Flutter 3.47 for the Android app (CI pins `3.47.x`)

### 1. Run the API

```bash
cd apps/api
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
# Let the website call the API from the browser. The API reads this from the shell, not from a .env file.
# Windows (PowerShell): $env:CORS_ORIGINS = "http://localhost:3000"
# Linux or macOS: export CORS_ORIGINS=http://localhost:3000
uvicorn app.main:app --port 8000
```

Open `http://localhost:8000/api/health`. When the model is loaded it answers with `"status": "ready"`. The first start downloads the ResNet-50 backbone weights.

Without `CORS_ORIGINS` the browser blocks the calls from the website, and the web app falls back to demo mode even though the API is running.

### 2. Run the website

```bash
cd apps/site
npm install
npm run dev
```

Open `http://localhost:3000`. Copy `.env.example` to `.env.local` first if you need to change the API address.

### 3. Run the Android app

```bash
cd apps/mobile
flutter pub get
flutter test
flutter run
```

The ONNX model is not committed. Export it from `apps/api/checkpoints/heads.pt` and place it at `apps/mobile/assets/spike/model_int8.onnx`.

### Environment variables

| App | Variable | Purpose |
|-----|----------|---------|
| API | `CORS_ORIGINS` | Comma-separated list of site origins allowed to call the API. Empty disables CORS. Read from the shell environment, not from a `.env` file. |
| API | `PORT` | Port to listen on. Set by Railway, defaults to `8000`. |
| Site | `NEXT_PUBLIC_API_URL` | Public URL of the API, used by the web app at `/app`. |
| Site | `GITHUB_REPO` | Repository whose releases feed the changelog, for example `alfzilham/BacteriaCV`. |
| Site | `GITHUB_WEBHOOK_SECRET` | Secret that verifies release webhooks from GitHub. |
| Site | `GITHUB_TOKEN` | Optional read token for a higher GitHub API rate limit. |

Examples live in `apps/api/.env.example` and `apps/site/.env.example`. The site reads `.env.local` on its own. The API does not read any `.env` file, so set its variables in the shell or in the Railway dashboard.

## Self-hosting

The website is static and can stay online without the API. The web app at `/app` needs an API to analyze images, so the hosted API may be switched off at times. To get the full experience, run your own.

**Website on Vercel**

1. Import the repository and set the root directory to `apps/site`.
2. Use the Next.js preset, install command `npm install`, build command `npm run build`.
3. Set `NEXT_PUBLIC_API_URL` to your API URL, plus `GITHUB_REPO` and `GITHUB_WEBHOOK_SECRET`.

**API on Railway**

1. Create a service from the repository and set the root directory to `apps/api`. `railway.json` already holds the start command and the health check (`/api/health`).
2. Set `CORS_ORIGINS` to your site domain.
3. Make sure the public domain targets the same port that the service listens on (the `PORT` value Railway sets). A mismatch shows up as a 502 error even though the service is online.

**Changelog webhook (optional)**

Add a GitHub webhook that points to `https://<your-site>/api/webhooks/github`, with content type `application/json`, the same secret as `GITHUB_WEBHOOK_SECRET`, and only the Releases event.

## API reference

Base path: `/api`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Service status and loaded checkpoint |
| `POST` | `/api/predict?stage=N` | Predict one image, `N` is the stage index from 0 to 4 |
| `GET` | `/api/report` | Latest evaluation report, or 404 if none exists |

Predict an image:

```bash
curl -X POST "http://localhost:8000/api/predict?stage=3" \
  -F "file=@apps/api/samples/contoh-1-bacilli-positif.png"
```

The response is JSON. The main fields are:

| Field | Description |
|-------|-------------|
| `prediction` | Shape and Gram labels with confidence values |
| `stage_index`, `stage_name`, `stage_title` | The selected stage |
| `stage_names` | The five stage keys in order |
| `panel` | The selected stage as a base64 PNG |
| `panels` | All five stages as base64 PNGs |
| `notes_text` | English notes about the result |
| `disclaimer` | The fixed low-confidence warning |
| `failed_stages` | Stages that could not be produced |

Errors return JSON with a `detail` field: `400` invalid file or stage, `413` file too large, `415` unsupported file type, `500` unexpected failure (generic message), `503` model not ready.

Four sample images are in `apps/api/samples/`.

## Testing and CI

```bash
# API
cd apps/api && python -m pytest tests -q

# Site
cd apps/site && npm run lint && npm test && npm run build

# Android
cd apps/mobile && flutter analyze --fatal-infos && flutter test

# Repository policy (no em-dashes or broken characters in text files)
bash tools/check-policy.sh
```

GitHub Actions runs these checks on every pull request: the API tests on Python 3.11, 3.12, and 3.13, the site build, Flutter analysis and tests, and the repository policy check.

## Releases and changelog

The changelog page reads GitHub Releases. Tag a web release with `web-v*` and an Android release with `mobile-v*`. When a release is published, the webhook refreshes the changelog on the site.

## Project status

- The website, the API, and the documentation are covered by CI.
- The Android app is still in its technical phase. See `apps/mobile/README.md` for what has been measured so far.
- Results are an aid for study and screening ideas. They are not a clinical tool.

## Contributing

Contributions are welcome. Please keep each pull request to one change.

1. Fork the repository and create a branch from the latest `main`.
2. Write commit messages in the Conventional Commits style, for example `fix: handle empty upload` or `docs: update quick start`.
3. Run the checks for the app you changed (see [Testing and CI](#testing-and-ci)).
4. Open a pull request and wait for the CI checks to turn green.

Do not change the checkpoint format, the endpoint contract, or the preprocessing stages without discussing it in an issue first, because the website, the API, and the Android app share the same model.

Found a bug or have a question? [Open an issue](https://github.com/alfzilham/BacteriaCV/issues).

## License

Released under the [MIT License](LICENSE).

Maintained by [Alfiz Ilham](https://github.com/alfzilham). The model is trained on the public DIBaS dataset. Built with PyTorch, FastAPI, Next.js, and Flutter.
