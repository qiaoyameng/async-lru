import asyncio

import pytest

from async_lru import AlruCacheQueueFull, alru_cache


class ConcurrencyTracker:
    def __init__(self) -> None:
        self.started = 0
        self.current = 0
        self.maximum = 0

    def enter(self) -> None:
        self.started += 1
        self.current += 1
        self.maximum = max(self.maximum, self.current)

    def exit(self) -> None:
        self.current -= 1


async def test_inflight_limit_caps_concurrent_executions() -> None:
    # maxsize=1, max_inflight=2: request several distinct keys, then
    # repeat one of them; the counter proves the business coroutine
    # never runs more than twice at the same time.
    tracker = ConcurrencyTracker()
    release = asyncio.Event()

    @alru_cache(maxsize=1, max_inflight=2)
    async def coro(val: int) -> int:
        tracker.enter()
        try:
            await release.wait()
            return val
        finally:
            tracker.exit()

    tasks = [asyncio.create_task(coro(k)) for k in (1, 2, 3, 4)]
    # repeat the key whose (still queued) entry is cached
    tasks.append(asyncio.create_task(coro(4)))

    for _ in range(3):
        await asyncio.sleep(0)

    # Only two business coroutines have been created so far; the
    # remaining calls are queued and their coroutines do not exist yet.
    assert tracker.started == 2
    assert tracker.current == 2

    release.set()
    results = await asyncio.gather(*tasks)

    assert results == [1, 2, 3, 4, 4]
    # Never more than two business coroutines executed at once.
    assert tracker.maximum == 2
    # The repeated key merged with the queued entry: 4 computations
    # for 5 callers.
    assert tracker.started == 4


async def test_same_key_merges_without_extra_slot() -> None:
    tracker = ConcurrencyTracker()
    release = asyncio.Event()

    @alru_cache(max_inflight=2)
    async def coro(val: int) -> int:
        tracker.enter()
        try:
            await release.wait()
            return val
        finally:
            tracker.exit()

    t1 = asyncio.create_task(coro(1))
    t2 = asyncio.create_task(coro(1))
    for _ in range(3):
        await asyncio.sleep(0)
    t3 = asyncio.create_task(coro(2))
    for _ in range(3):
        await asyncio.sleep(0)

    # key 1 merged into a single computation; key 2 took the 2nd slot
    assert tracker.started == 2
    assert tracker.current == 2

    release.set()
    assert await asyncio.gather(t1, t2, t3) == [1, 1, 2]
    assert tracker.maximum == 2


async def test_queue_full_raises_dedicated_error() -> None:
    release = asyncio.Event()

    @alru_cache(max_inflight=1, max_queue=1)
    async def coro(val: int) -> int:
        await release.wait()
        return val

    t1 = asyncio.create_task(coro(1))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(coro(2))
    await asyncio.sleep(0)

    with pytest.raises(AlruCacheQueueFull):
        await coro(3)

    release.set()
    assert await t1 == 1
    assert await t2 == 2


async def test_zero_queue_rejects_immediately() -> None:
    release = asyncio.Event()

    @alru_cache(max_inflight=1, max_queue=0)
    async def coro(val: int) -> int:
        await release.wait()
        return val

    t1 = asyncio.create_task(coro(1))
    await asyncio.sleep(0)

    with pytest.raises(AlruCacheQueueFull):
        await coro(2)

    release.set()
    assert await t1 == 1


async def test_evicted_task_still_holds_inflight_slot() -> None:
    # Cache eviction does not mean the task finished: an evicted but
    # still running task keeps its in-flight slot until completion.
    tracker = ConcurrencyTracker()
    release = asyncio.Event()

    @alru_cache(maxsize=1, max_inflight=2)
    async def coro(val: int) -> int:
        tracker.enter()
        try:
            await release.wait()
            return val
        finally:
            tracker.exit()

    t1 = asyncio.create_task(coro(1))
    t2 = asyncio.create_task(coro(2))
    t3 = asyncio.create_task(coro(3))
    for _ in range(3):
        await asyncio.sleep(0)

    # key 1 was evicted from the cache, but its task still runs and
    # still holds a slot, so key 3 has not started.
    assert not coro.cache_contains(1)
    assert tracker.started == 2

    release.set()
    assert await asyncio.gather(t1, t2, t3) == [1, 2, 3]
    assert tracker.started == 3
    assert tracker.maximum == 2


async def test_queued_entry_discarded_when_all_waiters_cancel() -> None:
    release = asyncio.Event()

    @alru_cache(max_inflight=1, max_queue=1)
    async def coro(val: int) -> int:
        await release.wait()
        return val

    t1 = asyncio.create_task(coro(1))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(coro(2))
    await asyncio.sleep(0)

    t2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t2
    await asyncio.sleep(0)

    # The queued entry was removed from the cache and the queue...
    assert not coro.cache_contains(2)

    # ...so a new key fits into the freed queue slot.
    t3 = asyncio.create_task(coro(3))
    await asyncio.sleep(0)
    with pytest.raises(AlruCacheQueueFull):
        await coro(4)

    release.set()
    assert await t1 == 1
    assert await t3 == 3


async def test_cache_close_rejects_queued_waiters() -> None:
    release = asyncio.Event()

    @alru_cache(max_inflight=1)
    async def coro(val: int) -> int:
        await release.wait()
        return val

    t1 = asyncio.create_task(coro(1))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(coro(2))
    await asyncio.sleep(0)

    close = asyncio.create_task(coro.cache_close(wait=True))
    await asyncio.sleep(0)

    # The queued waiter is rejected without ever starting computation.
    with pytest.raises(asyncio.CancelledError):
        await t2

    release.set()
    await close
    assert await t1 == 1


async def test_failed_task_releases_slot_for_queued() -> None:
    tracker = ConcurrencyTracker()

    @alru_cache(max_inflight=1)
    async def coro(val: int) -> int:
        tracker.enter()
        try:
            await asyncio.sleep(0.01)
            if val == 1:
                raise ValueError("boom")
            return val
        finally:
            tracker.exit()

    t1 = asyncio.create_task(coro(1))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(coro(2))

    with pytest.raises(ValueError, match="boom"):
        await t1
    # The failure callback released the slot and pumped the queue.
    assert await t2 == 2
    assert tracker.started == 2
    assert tracker.maximum == 1


async def test_cancelled_task_releases_slot_for_queued() -> None:
    tracker = ConcurrencyTracker()

    @alru_cache(max_inflight=1)
    async def coro(val: int) -> int:
        tracker.enter()
        try:
            await asyncio.sleep(5)
            return val
        finally:
            tracker.exit()

    t1 = asyncio.create_task(coro(1))
    for _ in range(2):
        await asyncio.sleep(0)
    t2 = asyncio.create_task(coro(2))
    await asyncio.sleep(0)

    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1

    # Cancellation cleanup released the slot, so key 2 may run now;
    # let it finish instead of waiting 5 seconds.
    for _ in range(5):
        await asyncio.sleep(0)
    assert tracker.started == 2
    t2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t2


async def test_invalid_inflight_queue_params() -> None:
    with pytest.raises(ValueError, match="max_inflight"):
        alru_cache(max_inflight=0)
    with pytest.raises(ValueError, match="max_queue requires max_inflight"):
        alru_cache(max_queue=1)
    with pytest.raises(ValueError, match="max_queue"):
        alru_cache(max_inflight=1, max_queue=-1)
