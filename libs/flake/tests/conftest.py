import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    # Mock async methods
    driver.launch_app = AsyncMock()

    # Mock sync methods
    driver.screenshot = MagicMock()
    driver.list_windows = MagicMock(return_value=[])

    return driver
