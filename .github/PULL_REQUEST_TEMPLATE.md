<!-- Add a description of your change here -->

Checklist (if not applicable, edit to add `(N/A)` and mark as done):

- [ ] If dependencies are added/removed: update `pyproject.toml` and `.pre-commit-config.yaml`.
- [ ] If new source files are added: add copyright notices dated with the current year.
- [ ] If qualification tests are changed: attach or link to a sample qualification report.
- [ ] If design has changed: ensure documentation is up to date.
- [ ] If ICD-defined sensors have been added: update `fake_servers.py` in katsdpcontroller to match.
- [ ] If the interface between the product controller and an engine has
      changed, update the VERSION constant for the engine (update the major
      number if sensors/requests have been removed/changed, minor number if
      only backwards-compatible changes have been done). Also update
      doc/control.rst with the new version.

Closes NGC-xxxx.
