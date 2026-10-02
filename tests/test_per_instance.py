import asyncio
import gc
import weakref

import pytest

from async_lru import alru_cache


class _Service:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls = 0

    @alru_cache(per_instance=True)
    async def get(self, value: int) -> tuple[str, int, int]:
        self.calls += 1
        await asyncio.sleep(0)
        return self.tag, value, self.calls


async def test_per_instance_independent_hits() -> None:
    first = _Service("a")
    second = _Service("b")

    assert await first.get(1) == ("a", 1, 1)
    assert await second.get(1) == ("b", 1, 1)
    # Second call per instance is served from its own cache.
    assert await first.get(1) == ("a", 1, 1)
    assert await second.get(1) == ("b", 1, 1)

    assert first.calls == 1
    assert second.calls == 1

    assert first.get.cache_info().hits == 1
    assert second.get.cache_info().hits == 1
    assert first.get.cache_info().currsize == 1
    assert second.get.cache_info().currsize == 1
    # Class-level access aggregates over live instances.
    info = _Service.get.cache_info()
    assert info.hits == 2
    assert info.misses == 2
    assert info.currsize == 2


async def test_concurrent_calls_hit_same_instance_task() -> None:
    started = asyncio.Event()
    finish = asyncio.Event()

    class Concurrent:
        calls = 0

        @alru_cache(per_instance=True)
        async def fetch(self, key: int) -> int:
            type(self).calls += 1
            started.set()
            await finish.wait()
            return key * 10

    obj = Concurrent()
    other = Concurrent()

    gather = asyncio.gather(obj.fetch(7), obj.fetch(7), other.fetch(7))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    finish.set()
    results = await gather

    assert results == [70, 70, 70]
    # Two callers on obj share one in-flight task; other has its own.
    assert Concurrent.calls == 2
    assert obj.fetch.cache_info().misses == 1
    assert obj.fetch.cache_info().hits == 1
    assert other.fetch.cache_info().misses == 1


async def test_close_one_instance_does_not_affect_other() -> None:
    class LocalService:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        @alru_cache(per_instance=True)
        async def get(self, value: int) -> tuple[str, int]:
            return self.tag, value

    first = LocalService("a")
    second = LocalService("b")

    await first.get(1)
    await second.get(1)

    await first.get.cache_close()

    assert first.get.cache_parameters()["closed"] is True
    assert second.get.cache_parameters()["closed"] is False

    with pytest.raises(RuntimeError):
        await first.get(2)
    assert await second.get(2) == ("b", 2)


async def test_idle_instance_is_garbage_collected() -> None:
    obj = _Service("a")
    await obj.get(1)

    ref = weakref.ref(obj)
    assert ref() is obj
    assert _Service.get.cache_info().currsize == 1

    del obj
    for _ in range(10):
        gc.collect()
        if ref() is None:
            break
        await asyncio.sleep(0)

    assert ref() is None
    # The dead instance's engine was removed from the manager registry.
    assert _Service.get.cache_info().currsize == 0


async def test_running_method_keeps_instance_until_done_then_releases() -> None:
    keep_running = asyncio.Event()

    class LongRunning:
        @alru_cache(per_instance=True)
        async def work(self) -> str:
            await keep_running.wait()
            return "done"

    obj = LongRunning()
    task = asyncio.create_task(obj.work())
    await asyncio.sleep(0)  # let the coroutine start

    ref = weakref.ref(obj)
    del obj
    gc.collect()
    # The running coroutine holds the only reference needed to finish.
    assert ref() is not None

    keep_running.set()
    assert await task == "done"

    for _ in range(10):
        gc.collect()
        if ref() is None:
            break
        await asyncio.sleep(0)

    # Completion must not resurrect/keep the instance.
    assert ref() is None


async def test_done_callback_does_not_resurrect_instance() -> None:
    class OneShot:
        @alru_cache(per_instance=True)
        async def ping(self) -> int:
            await asyncio.sleep(0)
            return 42

    obj = OneShot()
    assert await obj.ping() == 42

    ref = weakref.ref(obj)
    del obj
    for _ in range(10):
        gc.collect()
        await asyncio.sleep(0)  # let done callbacks run to completion
        if ref() is None:
            break

    assert ref() is None


def test_non_weakrefable_instance_raises_clear_error() -> None:
    class Solid:
        __slots__ = ()

        @alru_cache(per_instance=True)
        async def ping(self) -> None:
            ...

    with pytest.raises(TypeError, match="weak"):
        Solid().ping


def test_unhashable_instance_is_supported_without_id_reuse() -> None:
    class EqualUnhashable:
        __hash__ = None  # type: ignore[assignment]

        def __eq__(self, other: object) -> bool:
            return isinstance(other, EqualUnhashable)

        @alru_cache(per_instance=True)
        async def ping(self) -> int:
            return 1

    first = EqualUnhashable()
    second = EqualUnhashable()
    # The instances compare equal and are unhashable, but the
    # identity-keyed registry must still give them distinct caches.
    assert first == second
    with pytest.raises(TypeError):
        hash(first)
    assert first.ping is not second.ping


async def test_dead_instance_id_is_not_reused_for_cache() -> None:
    created: list[str] = []

    class Ephemeral:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        @alru_cache(per_instance=True)
        async def value(self) -> str:
            created.append(self.tag)
            return self.tag

    first = Ephemeral("first")
    first_addr = id(first)
    assert await first.value() == "first"
    assert first.value.cache_info().currsize == 1

    del first
    gc.collect()

    second = Ephemeral("second")
    # Whether or not the allocator reused the address, the new instance
    # must start with an empty cache.
    if id(second) == first_addr:
        assert second.value.cache_info().currsize == 0
    assert await second.value() == "second"
    assert second.value.cache_info().misses == 1
    assert created == ["first", "second"]


async def test_class_level_clear_and_invalidate() -> None:
    first = _Service("a")
    second = _Service("b")

    await first.get(1)
    await second.get(1)

    assert _Service.get.cache_invalidate(first, 1) is True
    assert first.get.cache_info().currsize == 0
    assert second.get.cache_info().currsize == 1

    _Service.get.cache_clear()
    assert second.get.cache_info().currsize == 0


async def test_class_level_close_closes_all_instances() -> None:
    class OwnService:
        @alru_cache(per_instance=True)
        async def get(self, value: int) -> int:
            return value

    first = OwnService()
    second = OwnService()

    await OwnService.get.cache_close()

    with pytest.raises(RuntimeError):
        await first.get(1)
    with pytest.raises(RuntimeError):
        await second.get(1)


async def test_class_direct_call_routes_to_instance_engine() -> None:
    obj = _Service("a")
    assert await _Service.get(obj, 5) == ("a", 5, 1)
    assert await _Service.get(obj, 5) == ("a", 5, 1)
    assert obj.get.cache_info().hits == 1


async def test_default_mode_still_shares_cache_keyed_by_instance() -> None:
    class Classic:
        @alru_cache
        async def get(self, value: int) -> int:
            return value

    obj = Classic()
    assert await Classic.get(obj, 1) == 1
    assert await obj.get(1) == 1
    assert Classic.get.cache_info().hits == 1


async def test_inflight_entry_removed_after_completion_when_instance_gone() -> None:
    gate = asyncio.Event()

    class Gated:
        @alru_cache(per_instance=True)
        async def work(self) -> int:
            await gate.wait()
            return 7

    obj = Gated()
    task = asyncio.create_task(obj.work())
    await asyncio.sleep(0)

    assert Gated.work.cache_info().currsize == 1

    ref = weakref.ref(obj)
    del obj
    gc.collect()
    # Still in flight: coroutine keeps the instance alive.
    assert ref() is not None

    gate.set()
    assert await task == 7

    for _ in range(10):
        gc.collect()
        await asyncio.sleep(0)
        if ref() is None:
            break

    assert ref() is None
    # The completion callback and the dead-instance finalizer must not
    # leave the engine (or its entry) behind.
    assert Gated.work.cache_info().currsize == 0


async def test_ttl_does_not_keep_instance_alive() -> None:
    class TTLService:
        @alru_cache(per_instance=True, ttl=100)
        async def get(self) -> int:
            return 1

    obj = TTLService()
    assert await obj.get() == 1

    ref = weakref.ref(obj)
    del obj
    for _ in range(10):
        gc.collect()
        if ref() is None:
            break
        await asyncio.sleep(0)

    assert ref() is None


async def test_class_management_without_instance_raises() -> None:
    with pytest.raises(TypeError, match="instance"):
        _Service.get.cache_contains(1)
    with pytest.raises(TypeError, match="instance"):
        _Service.get.cache_invalidate(1)
    with pytest.raises(TypeError, match="instance"):
        await _Service.get()
