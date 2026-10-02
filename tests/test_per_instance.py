import asyncio
import gc
import weakref

from functools import partialmethod

import pytest

from async_lru import alru_cache


class Client:
    def __init__(self, tag: str, event: asyncio.Event | None = None) -> None:
        self.tag = tag
        self.event = event or asyncio.Event()
        self.calls: list[tuple[str, int]] = []

    @alru_cache(per_instance=True)
    async def get(self, key: int) -> tuple[str, int]:
        self.calls.append((self.tag, key))
        await asyncio.sleep(0.05)
        return self.tag, key

    @alru_cache(per_instance=True)
    async def gated(self, key: int) -> tuple[str, int]:
        await self.event.wait()
        return self.tag, key


async def test_instances_have_independent_stats_and_entries() -> None:
    one = Client("one")
    two = Client("two")

    assert await one.get(1) == ("one", 1)
    assert await two.get(1) == ("two", 1)
    assert await one.get(1) == ("one", 1)

    assert one.calls == [("one", 1)]
    assert two.calls == [("two", 1)]

    info_one = one.get.cache_info()
    info_two = two.get.cache_info()
    assert info_one.hits == 1 and info_one.misses == 1
    assert info_two.hits == 0 and info_two.misses == 1
    assert info_one.currsize == 1 and info_two.currsize == 1

    assert one.get.cache_contains(1)
    assert not two.get.cache_contains(2)


async def test_concurrent_same_key_hits_per_instance() -> None:
    one = Client("one")
    two = Client("two")

    tasks = [
        asyncio.create_task(one.get(1)),
        asyncio.create_task(one.get(1)),
        asyncio.create_task(two.get(1)),
    ]
    results = await asyncio.gather(*tasks)

    assert results == [("one", 1), ("one", 1), ("two", 1)]
    # The wrapped coroutine runs once per instance even with concurrent calls.
    assert one.calls == [("one", 1)]
    assert two.calls == [("two", 1)]
    assert one.get.cache_parameters()["tasks"] == 0
    assert two.get.cache_parameters()["tasks"] == 0


async def test_closing_one_instance_keeps_others_usable() -> None:
    one = Client("one")
    two = Client("two")

    assert await one.get(1) == ("one", 1)
    await one.get.cache_close()

    with pytest.raises(RuntimeError):
        await one.get(2)

    assert one.get.cache_parameters()["closed"]
    assert not two.get.cache_parameters()["closed"]
    assert await two.get(1) == ("two", 1)


async def test_invalidation_is_scoped_to_instance() -> None:
    one = Client("one")
    two = Client("two")

    assert await one.get(1) == ("one", 1)
    assert await two.get(1) == ("two", 1)

    assert one.get.cache_invalidate(1)
    assert not one.get.cache_invalidate(1)
    assert two.get.cache_contains(1)

    assert await one.get(1) == ("one", 1)
    assert one.calls == [("one", 1), ("one", 1)]
    assert two.calls == [("two", 1)]


async def test_class_level_management_aggregates_live_instances() -> None:
    one = Client("one")
    two = Client("two")

    await one.get(1)
    await one.get(1)
    await two.get(1)

    info = Client.get.cache_info()
    assert (info.hits, info.misses, info.currsize) == (1, 2, 2)
    assert info.maxsize == 256

    Client.get.cache_clear()
    info = Client.get.cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)
    assert one.get.cache_info().currsize == 0

    # Explicit instance targeting from the class descriptor.
    await one.get(7)
    assert Client.get.cache_contains(one, 7)
    assert not Client.get.cache_contains(two, 7)
    assert Client.get.cache_invalidate(one, 7)
    assert not Client.get.cache_contains(one, 7)


async def test_idle_instance_is_garbage_collected() -> None:
    one = Client("one")
    ref = weakref.ref(one)

    await one.get(1)
    assert one.get.cache_contains(1)

    del one
    gc.collect()

    assert ref() is None


async def test_completed_callback_does_not_resurrect_instance() -> None:
    one = Client("one")
    ref = weakref.ref(one)
    event = one.event

    task = asyncio.create_task(one.gated(1))
    await asyncio.sleep(0)

    # The in-flight coroutine keeps the instance alive through the
    # normal bound-method reference even without an external variable.
    del one
    gc.collect()
    assert ref() is not None

    event.set()
    assert await task == ("one", 1)

    gc.collect()
    assert ref() is None


async def test_no_id_reuse_cache_cross_talk() -> None:
    class Reusable:
        @alru_cache(per_instance=True)
        async def get(self, key: int) -> int:
            return key * 2

    obj = Reusable()
    assert await obj.get(3) == 6
    del obj
    gc.collect()

    other = Reusable()
    assert not other.get.cache_contains(3)
    assert other.get.cache_info().currsize == 0
    assert await other.get(3) == 6
    assert other.get.cache_info().misses == 1


class NoWeakRef:
    __slots__ = ()

    @alru_cache(per_instance=True)
    async def get(self) -> int:
        return 1


class Unhashable:
    __hash__ = None  # type: ignore[assignment]

    @alru_cache(per_instance=True)
    async def get(self) -> int:
        return 1


async def test_unsupported_instances_raise_type_error() -> None:
    with pytest.raises(TypeError, match="weak references"):
        await NoWeakRef().get()

    with pytest.raises(TypeError, match="hashable"):
        await Unhashable().get()


async def test_inflight_close_cancels_only_target_instance() -> None:
    one = Client("one")
    two = Client("two")

    t1 = asyncio.create_task(one.get(1))
    t2 = asyncio.create_task(two.get(1))
    await asyncio.sleep(0)

    await one.get.cache_close(wait=False)

    with pytest.raises(asyncio.CancelledError):
        await t1
    assert await t2 == ("two", 1)


async def test_ttl_entries_are_discarded_when_instance_collected() -> None:
    class TimedClient:
        @alru_cache(per_instance=True, ttl=3600)
        async def get(self, key: int) -> int:
            return key

    client = TimedClient()
    ref = weakref.ref(client)
    assert await client.get(1) == 1
    assert client.get.cache_info().currsize == 1

    del client
    gc.collect()

    assert ref() is None
    # Let the event loop process any pending (already cancelled) timers.
    await asyncio.sleep(0)


async def test_per_instance_with_partialmethod() -> None:
    class Obj:
        async def _coro(self, val: int) -> int:
            return val

        coro = alru_cache(per_instance=True)(partialmethod(_coro, 2))

    obj = Obj()
    assert await obj.coro() == 2
    assert await obj.coro() == 2
    assert obj.coro.cache_info().hits == 1
