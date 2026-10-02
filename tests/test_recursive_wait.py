import asyncio

import pytest

from async_lru import (
    AlruCacheCall,
    AlruRecursionError,
    AlruRecursiveWaitError,
    alru_cache,
)


async def test_recursive_wait_same_key_fails_fast() -> None:
    started = asyncio.Event()

    @alru_cache(recursive_wait_detection=True)
    async def cached(value: str) -> str:
        if value == "outer":
            started.set()
            await asyncio.sleep(0)
            return await cached("inner")
        return await cached("outer")

    task = asyncio.create_task(cached("outer"))
    await started.wait()

    with pytest.raises(RecursionError) as exc_info:
        await task
    assert isinstance(exc_info.value, AlruRecursiveWaitError)

    assert [call.key for call in exc_info.value.chain] == [
        "outer",
        "inner",
        "outer",
    ]
    assert str(exc_info.value).count("cached") >= 3


async def test_recursive_wait_chain_locates_functions_and_keys() -> None:
    @alru_cache(recursive_wait_detection=True)
    async def fetch_a(value: str) -> str:
        return await fetch_b("shared")

    @alru_cache(recursive_wait_detection=True)
    async def fetch_b(value: str) -> str:
        return await fetch_a("shared")

    with pytest.raises(AlruRecursiveWaitError) as exc_info:
        await fetch_a("shared")

    functions = [
        call.function.rsplit(".", 1)[-1] for call in exc_info.value.chain
    ]
    assert functions == ["fetch_a", "fetch_b", "fetch_a"]
    assert [call.key for call in exc_info.value.chain] == [
        "shared",
        "shared",
        "shared",
    ]
    assert all(isinstance(call, AlruCacheCall) for call in exc_info.value.chain)


async def test_recursive_wait_alias() -> None:
    @alru_cache(detect_recursion=True)
    async def cached(value: str) -> str:
        if value == "outer":
            await asyncio.sleep(0)
            return await cached("inner")
        return await cached("outer")

    with pytest.raises(AlruRecursionError):
        await cached("outer")


async def test_independent_causal_chains_still_coalesce() -> None:
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    @alru_cache(recursive_wait_detection=True)
    async def cached(value: str) -> str:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return value

    first = asyncio.create_task(cached("shared"))
    second = asyncio.create_task(cached("shared"))
    await started.wait()
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(first, second) == ["shared", "shared"]
    assert calls == 1


async def test_non_cycle_recursive_chain_computes_once_per_key() -> None:
    calls = []

    @alru_cache(recursive_wait_detection=True)
    async def fetch_a(value: str) -> str:
        calls.append(("a", value))
        return await fetch_b("b")

    @alru_cache(recursive_wait_detection=True)
    async def fetch_b(value: str) -> str:
        calls.append(("b", value))
        return await fetch_c("c")

    @alru_cache(recursive_wait_detection=True)
    async def fetch_c(value: str) -> str:
        calls.append(("c", value))
        return value

    assert await fetch_a("a") == "c"
    assert calls == [("a", "a"), ("b", "b"), ("c", "c")]


async def test_legal_call_succeeds_after_recursive_failure() -> None:
    recurse = True

    @alru_cache(recursive_wait_detection=True)
    async def cached(value: str) -> str:
        if recurse and value == "outer":
            return await cached("inner")
        if recurse and value == "inner":
            return await cached("outer")
        return f"result:{value}"

    with pytest.raises(AlruRecursiveWaitError):
        await cached("outer")

    recurse = False
    assert await cached("outer") == "result:outer"
    assert await cached("inner") == "result:inner"


async def test_recursive_wait_detection_is_optional() -> None:
    @alru_cache
    async def cached(value: str) -> str:
        if value == "outer":
            await asyncio.sleep(0)
            return await cached("inner")
        return await cached("outer")

    task = asyncio.create_task(cached("outer"))
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_exception_cleans_recursive_call_state() -> None:
    calls = 0

    @alru_cache(recursive_wait_detection=True)
    async def cached(value: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("original error")
        return value

    with pytest.raises(RuntimeError, match="original error"):
        await cached("value")

    assert await cached("value") == "value"
    assert calls == 2


async def test_cancel_cleans_recursive_call_state() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    @alru_cache(recursive_wait_detection=True)
    async def cached(value: str) -> str:
        entered.set()
        await release.wait()
        return value

    task = asyncio.create_task(cached("value"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    await asyncio.sleep(0)
    assert await cached("value") == "value"


async def test_cancelling_one_waiter_keeps_shared_computation() -> None:
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    @alru_cache(recursive_wait_detection=True)
    async def cached(value: str) -> str:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return value

    owner = asyncio.create_task(cached("value"))
    await started.wait()
    waiter = asyncio.create_task(cached("value"))
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release.set()
    assert await owner == "value"
    assert calls == 1


async def test_eviction_deactivates_recursive_call() -> None:
    @alru_cache(
        maxsize=1,
        key=lambda depth, value: value,
        recursive_wait_detection=True,
    )
    async def cached(depth: int, value: str) -> str:
        if depth == 0:
            return await cached(1, "other")
        if depth == 1:
            return await cached(2, "start")
        return value

    assert await cached(0, "start") == "start"


async def test_key_diagnostic_is_bounded_and_safe() -> None:
    calls = 0

    class BadKey(str):
        def __repr__(self) -> str:
            return "k" * 10_000

    @alru_cache(recursive_wait_detection=True, key=lambda value: value)
    async def cached(value: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return await cached("inner")
        return await cached(BadKey("k" * 10_000))

    with pytest.raises(AlruRecursiveWaitError) as exc_info:
        await cached(BadKey("k" * 10_000))

    message = str(exc_info.value)
    assert len(message) < 1_000
    assert "<unrepresentable" in message
    assert "k" * 100 not in message


class RaisingReprKey(str):
    def __repr__(self) -> str:
        raise ValueError("diagnostic repr failed")


async def test_key_diagnostic_failure_does_not_mask_cycle() -> None:
    @alru_cache(recursive_wait_detection=True, key=lambda value: value)
    async def cached(value: str) -> str:
        if value == "inner":
            return await cached(RaisingReprKey("inner"))
        return await cached("inner")

    with pytest.raises(AlruRecursiveWaitError, match="unrepresentable"):
        await cached(RaisingReprKey("inner"))


async def test_original_error_is_not_masked_by_diagnostics() -> None:
    @alru_cache(recursive_wait_detection=True)
    async def cached(value: str) -> str:
        if value == "start":
            raise ValueError("original error")
        return value

    with pytest.raises(ValueError, match="original error"):
        await cached("start")
