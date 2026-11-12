# Contributing to BacteriaCV

Thanks for your interest. Contributions are welcome: bug reports, fixes, documentation, and translations.

BacteriaCV is a monorepo with three apps that share one model: the website (`apps/site`), the inference API (`apps/api`), and the Android app (`apps/mobile`). See the [README](README.md) for the layout and for how to run each app.

## Before you start

- For a bug or a small fix, open a pull request directly.
- For anything bigger, [open an issue](https://github.com/alfzilham/BacteriaCV/issues) first so we can agree on the approach.
- Do not change the checkpoint format (`apps/api/checkpoints/heads.pt`), the endpoint contract of the API, or the preprocessing stages without discussing it in an issue. The website, the API, and the Android app share the same model, so a change in one place breaks the others.

## Set up

1. Fork the repository and create a branch from the latest `main`.
2. Run the app you are changing. The steps are in the README under "Getting started", and the website has a guide at `/docs/self-host`.
3. When the website runs next to a local API, set `CORS_ORIGINS` in the shell that starts the API. The API reads it from the shell, not from a `.env` file.

## Make a change

- Keep each pull request to one change. Do not mix a refactor with a new feature.
- Write commit messages in the Conventional Commits style, for example `fix: handle empty upload` or `docs: update quick start`.
- Interface text lives in `apps/site/src/models/copy.ts`. Every entry is a pair, English first and Indonesian second. Add both.
- Do not use the em-dash character (U+2014) in any text file. The policy check rejects it.

## Run the checks

Run the checks for every app you changed:

```bash
# API
cd apps/api && python -m pytest tests -q

# Website
cd apps/site && npm run lint && npm test && npm run build

# Android
cd apps/mobile && flutter analyze --fatal-infos && flutter test

# Repository policy (no em-dashes or broken characters in text files)
bash tools/check-policy.sh
```

GitHub Actions runs the same checks on every pull request: the API tests on Python 3.11, 3.12, and 3.13, the website build, Flutter analysis and tests, and the policy check. A pull request can be merged once they are green.

## Open a pull request

1. Push your branch and open a pull request against `main`.
2. Fill in the pull request template: what changed, how you checked it, and anything that needs a manual step.
3. Link the issue it closes, for example `Closes #12`.

## Security

Do not report security problems in a public issue. See [SECURITY.md](SECURITY.md).

## License

By contributing you agree that your work is released under the [MIT License](LICENSE).
