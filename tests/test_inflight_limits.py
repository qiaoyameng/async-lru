import asyncio
from collections.abc import Callable

import pytest

from async_lru import ALRU_QUEUE_FULL, alru_cache


async def test_inflight_limit_and_queue_limit_with_counter() -> None:
    active = 0
    max_active = 0
    started: dict[int, asyncio.Event] = {}
    release: dict[int, asyncio.Event] = {}

    @alru_cache(maxsize=1, inflight_limit=2, queue_limit=2)
    async def coro(value: int) -> int:
        nonlocal active, max_active

        assert len(coro._LRUCacheWrapper__inflight) <= 2  # type: ignore[attr-defined]
        active += 1
        max_active = max(max_active, active)
        started[value].set()
        await release[value].wait()
        active -= 1
        return value

    values = [1, 2, 3, 4, 5, 1]
    for value in values:
        started[value] = asyncio.Event()
        release[value] = asyncio.Event()

    tasks = [asyncio.create_task(coro(value)) for value in values]
    await asyncio.gather(started[1].wait(), started[2].wait())
    await asyncio.sleep(0)

    assert max_active == 2
    await asyncio.sleep(0)
    assert tasks[4].result() is ALRU_QUEUE_FULL
    assert not tasks[5].done()
    assert coro.cache_info().hits == 1
    assert coro.cache_info().currsize == 1
    parameters = coro.cache_parameters()
    assert parameters["inflight_limit"] == 2
    assert parameters["queue_limit"] == 2
    assert parameters["tasks"] == 2

    release[1].set()
    await started[3].wait()
    assert max_active == 2

    release[2].set()
    await started[4].wait()
    assert max_active == 2

    release[3].set()
    release[4].set()

    results = await asyncio.gather(*tasks[:4], tasks[5])
    assert results == [1, 2, 3, 4, 1]
    assert max_active == 2


async def test_same_key_in_queue_merges_work() -> None:
    calls = 0
    gate = asyncio.Event()

    @alru_cache(inflight_limit=1, queue_limit=1)
    async def coro(value: int) -> int:
        nonlocal calls
        calls += 1
        await gate.wait()
        return value

    first = asyncio.create_task(coro(1))
    queued1 = asyncio.create_task(coro(2))
    queued2 = asyncio.create_task(coro(2))
    rejected = asyncio.create_task(coro(3))

    await asyncio.sleep(0)
    assert queued1.done() is False
    assert queued2.done() is False
    assert rejected.result() is ALRU_QUEUE_FULL
    assert coro.cache_info().hits == 1

    gate.set()
    assert await asyncio.gather(first, queued1, queued2) == [1, 2, 2]
    assert calls == 2


async def test_all_waiters_cancelled_remove_queued_work() -> None:
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    @alru_cache(inflight_limit=1, queue_limit=1)
    async def coro(value: int) -> int:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return value

    active = asyncio.create_task(coro(1))
    queued = asyncio.create_task(coro(2))
    await started.wait()
    await asyncio.sleep(0)

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    release.set()
    assert await active == 1
    assert calls == 1

    assert await coro(2) == 2
    assert calls == 2


async def test_cache_invalidate_cancels_queued_work() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    @alru_cache(inflight_limit=1, queue_limit=1)
    async def coro(value: int) -> int:
        started.set()
        await release.wait()
        return value

    active = asyncio.create_task(coro(1))
    queued = asyncio.create_task(coro(2))
    await started.wait()
    await asyncio.sleep(0)

    assert coro.cache_invalidate(2)
    with pytest.raises(asyncio.CancelledError):
        await queued

    release.set()
    assert await active == 1
    assert await coro(2) == 2


async def test_failed_task_releases_inflight_and_promotes_queue() -> None:
    active = 0
    max_active = 0
    started: list[asyncio.Event] = []
    release: list[asyncio.Event] = []

    @alru_cache(inflight_limit=2, queue_limit=1)
    async def coro(value: int) -> int:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        started[value].set()
        await release[value].wait()
        active -= 1
        if value == 0:
            raise RuntimeError("boom")
        return value

    for _ in range(3):
        started.append(asyncio.Event())
        release.append(asyncio.Event())

    failed = asyncio.create_task(coro(0))
    active2 = asyncio.create_task(coro(1))
    queued = asyncio.create_task(coro(2))
    await asyncio.gather(started[0].wait(), started[1].wait())
    await asyncio.sleep(0)
    assert max_active == 2

    release[0].set()
    with pytest.raises(RuntimeError):
        await failed
    await started[2].wait()
    assert max_active == 2

    release[1].set()
    release[2].set()
    assert await asyncio.gather(active2, queued) == [1, 2]


async def test_synchronous_error_during_promotion_unblocks_queue() -> None:
    release_first = asyncio.Event()

    async def coro(value: int) -> int:
        if value == 2:
            raise RuntimeError("before first await")
        await release_first.wait()
        return value

    cached = alru_cache(inflight_limit=1, queue_limit=2)(coro)
    first = asyncio.create_task(cached(1))
    failed = asyncio.create_task(cached(2))
    third = asyncio.create_task(cached(3))

    await asyncio.sleep(0)
    release_first.set()

    assert await first == 1
    with pytest.raises(RuntimeError, match="before first await"):
        await failed
    assert await third == 3


async def test_cache_close_cancels_queued_work(check_lru: Callable[..., None]) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    @alru_cache(inflight_limit=1, queue_limit=1)
    async def coro(value: int) -> int:
        started.set()
        await release.wait()
        return value

    active = asyncio.create_task(coro(1))
    queued = asyncio.create_task(coro(2))
    await started.wait()
    await asyncio.sleep(0)

    close = asyncio.create_task(coro.cache_close())
    await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await queued
    with pytest.raises(asyncio.CancelledError):
        await active
    await close
    release.set()
    check_lru(coro, hits=0, misses=2, cache=0, tasks=0)


async def test_cache_close_wait_drains_queued_work() -> None:
    started = asyncio.Event()

    @alru_cache(inflight_limit=1, queue_limit=1)
    async def coro(value: int) -> int:
        started.set()
        await asyncio.sleep(0.01)
        return value

    active = asyncio.create_task(coro(1))
    queued = asyncio.create_task(coro(2))
    await started.wait()

    await coro.cache_close(wait=True)
    assert await asyncio.gather(active, queued) == [1, 2]


@pytest.mark.parametrize(
    ("inflight_limit", "queue_limit", "message"),
    (
        (0, None, "inflight_limit must be a positive integer"),
        (1, -1, "queue_limit must be a non-negative integer"),
        (True, None, "inflight_limit must be a positive integer"),
    ),
)
async def test_invalid_limits(
    inflight_limit: int, queue_limit: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        alru_cache(inflight_limit=inflight_limit, queue_limit=queue_limit)


async def test_queue_limit_requires_inflight_limit() -> None:
    with pytest.raises(ValueError, match="queue_limit requires inflight_limit"):
        alru_cache(queue_limit=1)
