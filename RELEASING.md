# Releasing q-alchemy-sdk-py

Publishing to [PyPI](https://pypi.org/project/q-alchemy-sdk-py/) is automated via
GitHub Actions and [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) — there are no API tokens to manage or rotate.

## One-time setup (already done? check before repeating)

1. **PyPI** — as an owner of the `q-alchemy-sdk-py` project: *Manage → Publishing →
   Add a new publisher* with
   - Owner: `data-cybernetics`
   - Repository: `q-alchemy-sdk-py`
   - Workflow name: `publish.yaml`
   - Environment name: `pypi`
2. **GitHub** — *Settings → Environments → New environment* named `pypi`.
   Optionally add *required reviewers* there: publishing then pauses for an
   explicit approval click, mirroring the manual-gate policy used on the
   internal Gitea pipelines.

## Release process

1. Bump `[project].version` in `pyproject.toml` (PR, review, merge to `main`).
2. Create a **GitHub Release** with tag `v<version>` (e.g. `v0.2.28`) on `main`.
3. The `Publish to PyPI` workflow builds sdist + wheel, verifies the tag matches
   the package version, smoke-imports the wheel, and publishes.

Notes:

- A tag/version mismatch fails the build step on purpose — fix the tag or the
  version, never force.
- PyPI rejects re-uploads of an existing version; a botched release needs a new
  patch version (yank the bad one on PyPI if necessary).
- The integration tests are *live-service* tests requiring `Q_ALCHEMY_API_KEY`
  and are intentionally not part of the publish workflow — run them before
  merging release PRs.
