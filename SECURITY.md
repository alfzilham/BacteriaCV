# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately by email to alfizilham@gmail.com, with the subject "BacteriaCV security". Include:

- what is affected (website, API, or Android app) and the version or commit;
- the steps to reproduce it;
- what you think the impact is.

Use GitHub's private vulnerability reporting instead if it is enabled on the repository (Security tab, "Report a vulnerability").

## Supported versions

Only the latest release and the current `main` branch receive fixes.

## Scope and notes for people who deploy it

- The API accepts image uploads of up to 20 MB and runs the model on the CPU. It limits `POST /api/predict` to 20 calls per client per minute by default (`RATE_LIMIT_PER_MINUTE`, `0` turns it off). The counter is kept in memory per process, so it does not replace a limit at your proxy or hosting platform if you run several replicas or expect abuse.
- Set `CORS_ORIGINS` to your website address only. An empty value turns CORS off.
- Keep `GITHUB_WEBHOOK_SECRET` and `GITHUB_TOKEN` out of the repository. Set them in your hosting dashboard. The webhook rejects any request whose signature does not match the secret.
- The API does not store uploaded images.

## Not a medical device

BacteriaCV is an aid for study, not a diagnosis. A wrong prediction is a model limitation, not a security issue. Report it as a normal bug.
