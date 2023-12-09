# noqa: D104

import pytest

# Make the event_loop package-scoped
pytestmark = pytest.mark.asyncio(scope="package")
