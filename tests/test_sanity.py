import aiohttp


def test_aiohttp_importable():
    assert aiohttp.__version__


async def test_asyncio_event_loop_runs():
    assert True
