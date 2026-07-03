import pytest
import asyncio
from unittest.mock import patch
import time
from PIL import Image

from replay.waits import wait_for_file, wait_for_screenshot_stable


@pytest.mark.asyncio
async def test_wait_for_file_succeeds(tmp_path):
    file_path = tmp_path / "test.txt"

    async def create_file_later():
        await asyncio.sleep(0.01)
        file_path.touch()

    task1 = asyncio.create_task(create_file_later())
    task2 = asyncio.create_task(wait_for_file(file_path, timeout=1.0, poll=0.01))

    await asyncio.gather(task1, task2)
    assert file_path.exists()


@pytest.mark.asyncio
@patch("asyncio.sleep")
async def test_wait_for_file_timeout(mock_sleep, tmp_path):
    file_path = tmp_path / "missing.txt"

    # Fast forward time to force a timeout exception without waiting real time
    with patch("time.monotonic", side_effect=[0, 0.1, 10.1]):
        with pytest.raises(TimeoutError) as exc:
            await wait_for_file(file_path, timeout=5.0, poll=1.0)

    assert "Timed out waiting for file" in str(exc.value)
    assert "missing.txt" in str(exc.value)


@pytest.mark.asyncio
@patch("asyncio.sleep")
async def test_wait_for_screenshot_stable(mock_sleep, mock_driver):
    # Mock driver.screenshot to return unstable images then stable
    img1 = Image.new("RGB", (100, 100), color="white")
    img2 = Image.new("RGB", (100, 100), color="black")
    img3 = Image.new("RGB", (100, 100), color="black")

    mock_driver.screenshot.side_effect = [img1, img2, img3]

    with patch("time.monotonic", side_effect=[0, 0.1, 0.2, 0.3]):
        await wait_for_screenshot_stable(mock_driver, timeout=1.0, poll=0.1)

    assert mock_driver.screenshot.call_count == 3
    # Called twice in between the 3 screenshot polls
    assert mock_sleep.call_count == 2
