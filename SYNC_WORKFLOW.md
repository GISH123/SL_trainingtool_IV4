# External ↔ Internal synchronization workflow

## Canonical rule

The external GitHub repository is the canonical source tree for the IV4 training tool.
The internal/offline repository is a controlled mirror updated through verified `git bundle` files.

Once established, do not manually copy individual source files between environments.

## Commit to Git

Commit source code, BAT/PowerShell helpers, tests, README/reconstruction notes,
requirements files, and small non-secret configuration templates.

Do not commit datasets, labels, checkpoints, pretrained weights, TFRecord files,
generated reports/snapshots, secrets, `.env`, internal credentials, or transfer bundles.

## External development

1. `git status --short --branch`
2. Develop and test.
3. `git add <explicit files>`
4. `git commit -m "<message>"`
5. `git push origin main`
6. `powershell -ExecutionPolicy Bypass -File .\tools\export_bundle.ps1`
7. Transfer the generated `.bundle` to the internal environment.

## Internal update

Dry-run first:

`powershell -ExecutionPolicy Bypass -File .\tools\import_bundle.ps1 -BundlePath D:\path\to\update.bundle`

After reviewing incoming commits and diff:

`powershell -ExecutionPolicy Bypass -File .\tools\import_bundle.ps1 -BundlePath D:\path\to\update.bundle -Apply`

The script uses `git merge --ff-only`. If internal and external diverged, it stops.

## Internal hotfix rule

Avoid internal-only source edits.

If an emergency internal code change is unavoidable:
1. Commit it on `internal/hotfix-YYYYMMDD`.
2. Export that branch as a reverse bundle.
3. Bring the bundle back outside.
4. Integrate it into GitHub before continuing normal development.

## Sync identity

External:
`git rev-parse HEAD`

Internal:
`git rev-parse HEAD`

If both SHAs are identical and both working trees are clean, source code is synchronized.


## One-time internal bootstrap

The first internal copy must share the same Git history as the external repository.
Do not initialize an unrelated Git history inside the old internal folder.

After the external baseline commit is pushed, export the first bundle and transfer it inside.

Create a new internal mirror:

`powershell -ExecutionPolicy Bypass -File .\tools\bootstrap_internal_from_bundle.ps1 -BundlePath D:\transfer\SL_trainingtool_IV4_main_<stamp>_<sha>.bundle -Destination D:\path\SL_trainingtool_IV4`

Then copy only ignored local runtime/data artifacts from the old internal folder into the new mirror
(for example datasets, checkpoints, model weights, reports, and snapshots).

Verify:

`git rev-parse HEAD`

The external and internal HEAD SHA must be identical before normal bundle updates begin.
