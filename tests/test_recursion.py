import asyncio

import pytest

from async_lru import AlruCacheRecursionError, alru_cache


async def test_self_recursion_same_key_fails_fast() -> None:
    @alru_cache(detect_recursion=True)
    async def coro(val: int) -> int:
        return await coro(val)

    with pytest.raises(AlruCacheRecursionError):
        await coro(1)


async def test_cycle_across_two_decorators_fails_fast() -> None:
    @alru_cache(detect_recursion=True)
    async def func_a(val: int) -> int:
        return await func_b(val)

    @alru_cache(detect_recursion=True)
    async def func_b(val: int) -> int:
        return await func_a(val)

    with pytest.raises(AlruCacheRecursionError):
        await func_a(1)


async def test_error_message_locates_functions_and_keys() -> None:
    @alru_cache(detect_recursion=True)
    async def func_a(val: int) -> int:
        return await func_b(val + 1)

    @alru_cache(detect_recursion=True)
    async def func_b(val: int) -> int:
        return await func_a(val - 1)

    with pytest.raises(AlruCacheRecursionError) as exc_info:
        await func_a(1)

    message = str(exc_info.value)
    assert "func_a" in message
    assert "func_b" in message
    assert "key=1" in message
    assert "key=2" in message
    # The chain ends where it started, showing the closing of the loop.
    assert message.index("func_a") < message.index("func_b")


async def test_chain_without_cycle_computes_each_once() -> None:
    counts = {"a": 0, "b": 0, "c": 0}

    @alru_cache(detect_recursion=True)
    async def func_a(val: int) -> int:
        counts["a"] += 1
        return await func_b(val) + 1

    @alru_cache(detect_recursion=True)
    async def func_b(val: int) -> int:
        counts["b"] += 1
        return await func_c(val) + 1

    @alru_cache(detect_recursion=True)
    async def func_c(val: int) -> int:
        counts["c"] += 1
        return val + 1

    assert await func_a(1) == 4
    assert counts == {"a": 1, "b": 1, "c": 1}


async def test_concurrent_independent_callers_still_merge() -> None:
    calls = 0

    @alru_cache(detect_recursion=True)
    async def coro(val: int) -> int:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return val * 2

    results = await asyncio.gather(coro(2), coro(2), coro(2))

    assert results == [4, 4, 4]
    assert calls == 1


async def test_same_key_gathered_inside_chain_still_merges() -> None:
    calls = 0

    @alru_cache(detect_recursion=True)
    async def coro(val: int) -> int:
        nonlocal calls
        calls += 1
        if val == 2:
            first, second = await asyncio.gather(coro(1), coro(1))
            return first + second
        await asyncio.sleep(0.01)
        return val

    assert await coro(2) == 2
    assert calls == 2


async def test_legal_call_succeeds_after_recursion_error() -> None:
    @alru_cache(detect_recursion=True)
    async def func_a(val: int) -> int:
        if val == 0:
            return await func_b(val)
        return val * 10

    @alru_cache(detect_recursion=True)
    async def func_b(val: int) -> int:
        return await func_a(val)

    with pytest.raises(AlruCacheRecursionError):
        await func_a(0)

    # The failed entries were dropped; unrelated keys work fine.
    assert await func_a(5) == 50
    # The cyclic path is still detected on retry (no leaked chain state).
    with pytest.raises(AlruCacheRecursionError):
        await func_a(0)


async def test_chain_state_cleaned_after_cancellation() -> None:
    calls = 0

    @alru_cache(detect_recursion=True)
    async def coro(val: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(10)
        return val

    task = asyncio.create_task(coro(1))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await coro(1) == 1


async def test_cycle_detected_even_after_eviction() -> None:
    @alru_cache(maxsize=1, detect_recursion=True)
    async def func_a(val: int) -> int:
        return await func_b(val)

    @alru_cache(maxsize=1, detect_recursion=True)
    async def func_b(val: int) -> int:
        return await func_a(val)

    with pytest.raises(AlruCacheRecursionError):
        await func_a(1)


async def test_recursive_memoization_has_no_false_positive() -> None:
    @alru_cache(detect_recursion=True)
    async def fib(n: int) -> int:
        if n < 2:
            return n
        return await fib(n - 1) + await fib(n - 2)

    assert await fib(10) == 55


async def test_key_repr_in_error_is_bounded() -> None:
    class BigKey:
        def __repr__(self) -> str:
            return "x" * 1_000_000

        def __hash__(self) -> int:
            return hash("big")

        def __eq__(self, other: object) -> bool:
            return self is other

    @alru_cache(detect_recursion=True)
    async def coro(val: object) -> object:
        return await coro(val)

    with pytest.raises(AlruCacheRecursionError) as exc_info:
        await coro(BigKey())

    assert len(str(exc_info.value)) < 1000


async def test_detection_disabled_by_default_still_caches() -> None:
    calls = 0

    @alru_cache
    async def coro(val: int) -> int:
        nonlocal calls
        calls += 1
        return val

    assert await coro(1) == 1
    assert await coro(1) == 1
    assert calls == 1
