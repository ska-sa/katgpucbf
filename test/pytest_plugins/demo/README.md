# Demonstration of pytest plugins

This directory contains an example of using the `katgpucbf.pytest_plugins`
package. It is useful for two purposes:

1. It can be run manually with `pytest test/pytest_plugins/demo/demo.py`.
   This will produce a `report.json` in the current directory and a
   `arrays` subdirectory of the directory containing this README. The
   `report.json` can be processed into a PDF using
   `qualification/report/generate_pdf.py`. The `arrays` can be manually
   inspected to check that the failing numpy arrays are correctly written.
   Note that some of the tests are *expected* to fail.

2. It is run automatically by unit tests in `test/pytest_plugins` to verify
   that the plugins function correctly.
