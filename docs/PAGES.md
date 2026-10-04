# Project page deployment

The static project page lives in `site/`; research documentation remains in `docs/`.

In repository Settings → Pages → Build and deployment, select **Source: GitHub Actions**. No branch/folder selector is used for this mode. The `Deploy project page` workflow runs from `main` and publishes only `site/`. If needed, run it manually from the Actions tab after changing the setting.

Expected project URL: https://jinsuby.github.io/ProactiveCoach/

All local assets and the annotation JSON use relative paths, so the page works beneath `/ProactiveCoach/`. No build tool, external font, or analytics service is required. No `.openai` hosting metadata is included.

The page preserves the supplied project-page content, figures, examples, and manuscript-result caveats. The code README retains its separate reproducibility limits.
