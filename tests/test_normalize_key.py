from collections.abc import Callable

import pytest

from async_lru import alru_cache


async def test_normalize_key_equivalent_spellings(
    check_lru: Callable[..., None],
) -> None:
    """f(1), f(x=1) and f(1, mode="a") share one cache entry."""
    calls = 0

    @alru_cache(normalize_key=True)
    async def fetch(host: str, port: int = 8080, mode: str = "a") -> int:
        nonlocal calls
        calls += 1
        return calls

    equivalent = [
        (("h",), {}),
        (("h", 8080), {}),
        (("h", 8080, "a"), {}),
        (("h", 8080), {"mode": "a"}),
        (("h",), {"port": 8080}),
        (("h",), {"mode": "a"}),
        ((), {"host": "h", "port": 8080, "mode": "a"}),
    ]
    for i, (args, kwargs) in enumerate(equivalent):
        assert await fetch(*args, **kwargs) == 1
        check_lru(fetch, hits=i, misses=1, cache=1, tasks=0)


async def test_normalize_key_non_equivalent_spellings(
    check_lru: Callable[..., None],
) -> None:
    @alru_cache(normalize_key=True)
    async def fetch(host: str, port: int = 8080, mode: str = "a") -> str:
        return f"{host}:{port}:{mode}"

    assert await fetch("h") == "h:8080:a"
    assert await fetch("h", 8081) == "h:8081:a"
    assert await fetch("h", mode="b") == "h:8080:b"
    assert await fetch("x") == "x:8080:a"
    check_lru(fetch, hits=0, misses=4, cache=4, tasks=0)

    # Each non-default spelling now hits its own entry.
    assert await fetch("h", port=8081) == "h:8081:a"
    assert await fetch("h", 8080, "b") == "h:8080:b"
    check_lru(fetch, hits=2, misses=4, cache=4, tasks=0)


async def test_normalize_key_positional_only_and_keyword_only(
    check_lru: Callable[..., None],
) -> None:
    @alru_cache(normalize_key=True)
    async def coro(a: int, /, b: int, *, c: int = 3) -> int:
        return a * 100 + b * 10 + c

    assert await coro(1, 2) == 123
    assert await coro(1, b=2) == 123
    assert await coro(1, 2, c=3) == 123
    assert await coro(1, b=2, c=3) == 123
    check_lru(coro, hits=3, misses=1, cache=1, tasks=0)

    assert await coro(1, 2, c=4) == 124
    check_lru(coro, hits=3, misses=2, cache=2, tasks=0)

    # A positional-only parameter passed by keyword is an invalid call.
    with pytest.raises(TypeError):
        await coro(a=1, b=2)  # type: ignore[call-arg,misc]


async def test_normalize_key_var_positional(check_lru: Callable[..., None]) -> None:
    @alru_cache(normalize_key=True)
    async def coro(*args: int) -> int:
        return len(args)

    assert await coro(1, 2) == 2
    assert await coro(1, 2) == 2
    check_lru(coro, hits=1, misses=1, cache=1, tasks=0)

    assert await coro(2, 1) == 2
    assert await coro() == 0
    check_lru(coro, hits=1, misses=3, cache=3, tasks=0)


async def test_normalize_key_var_keyword(check_lru: Callable[..., None]) -> None:
    """**kwargs flatten into the key; keyword order is normalized."""

    @alru_cache(normalize_key=True)
    async def coro(a: int, b: int = 2, **kw: int) -> tuple[int, int, int]:
        return (a, b, sum(kw.values()))

    assert await coro(1, x=5) == (1, 2, 5)
    assert await coro(1, 2, x=5) == (1, 2, 5)
    assert await coro(a=1, b=2, **{"x": 5}) == (1, 2, 5)
    check_lru(coro, hits=2, misses=1, cache=1, tasks=0)

    # Same mapping, different keyword order: still one entry.
    assert await coro(1, x=1, y=2) == (1, 2, 3)
    assert await coro(1, y=2, x=1) == (1, 2, 3)
    check_lru(coro, hits=3, misses=2, cache=2, tasks=0)

    # An extra kwarg is a different call.
    assert await coro(1, x=5, y=6) == (1, 2, 11)
    check_lru(coro, hits=3, misses=3, cache=3, tasks=0)


async def test_normalize_key_bound_method(check_lru: Callable[..., None]) -> None:
    class Service:
        @alru_cache(normalize_key=True)
        async def get(self, key: str, mode: str = "r") -> str:
            return f"{id(self)}:{key}:{mode}"

    svc = Service()
    result = await svc.get("k")
    assert await svc.get("k", mode="r") == result
    assert await svc.get(key="k") == result
    check_lru(Service.get, hits=2, misses=1, cache=1, tasks=0)

    # The instance is part of the key: another instance misses.
    other = Service()
    assert await other.get("k") != result
    check_lru(Service.get, hits=2, misses=2, cache=2, tasks=0)

    # Bound cache_invalidate/cache_contains normalize the same way.
    assert svc.get.cache_invalidate("k", mode="r")
    assert not svc.get.cache_contains("k")
    assert other.get.cache_contains("k")
    check_lru(Service.get, hits=2, misses=2, cache=1, tasks=0)


async def test_normalize_key_invalid_calls_raise_type_error(
    check_lru: Callable[..., None],
) -> None:
    @alru_cache(normalize_key=True)
    async def fetch(host: str, port: int = 8080) -> str:
        return f"{host}:{port}"  # pragma: no cover  # never reached here

    with pytest.raises(TypeError):
        await fetch()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await fetch("h", 1, "extra")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await fetch("h", bogus=1)  # type: ignore[call-arg]

    # The wrapped function raises TypeError for the same calls.
    with pytest.raises(TypeError):
        await fetch.__wrapped__()  # type: ignore[call-arg]

    # Failed calls neither populate the cache nor count as misses.
    check_lru(fetch, hits=0, misses=0, cache=0, tasks=0)


async def test_normalize_key_typed_separates_bool_and_int(
    check_lru: Callable[..., None],
) -> None:
    @alru_cache(normalize_key=True, typed=True)
    async def coro(val: int = 0) -> int:
        return val

    assert await coro(1) == 1
    assert await coro(True) is True
    check_lru(coro, hits=0, misses=2, cache=2, tasks=0)
    assert coro.cache_contains(1)
    assert coro.cache_contains(True)


async def test_normalize_key_untyped_merges_bool_and_int(
    check_lru: Callable[..., None],
) -> None:
    @alru_cache(normalize_key=True)
    async def coro(val: int = 0) -> int:
        return val

    assert await coro(1) == 1
    assert await coro(True) == 1  # same entry, like functools.lru_cache
    check_lru(coro, hits=1, misses=1, cache=1, tasks=0)


async def test_normalize_key_unhashable_value_raises(
    check_lru: Callable[..., None],
) -> None:
    @alru_cache(normalize_key=True)
    async def coro(val: list[int]) -> int:
        return len(val)  # pragma: no cover  # the key fails before the call

    with pytest.raises(TypeError, match="unhashable"):
        await coro([1, 2])
    check_lru(coro, hits=0, misses=0, cache=0, tasks=0)


async def test_normalize_key_mutable_default_not_stringified(
    check_lru: Callable[..., None],
) -> None:
    """A mutable default must not silently become a stable string key."""

    @alru_cache(normalize_key=True)
    async def coro(val: list[int] = []) -> int:  # noqa: B006
        return len(val)  # pragma: no cover  # the key fails before the call

    with pytest.raises(TypeError, match="unhashable"):
        await coro()
    check_lru(coro, hits=0, misses=0, cache=0, tasks=0)


def test_normalize_key_with_custom_key_raises() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        alru_cache(key=lambda x: x, normalize_key=True)


async def test_normalize_key_cache_invalidate_across_spellings(
    check_lru: Callable[..., None],
) -> None:
    """cache_invalidate with any equivalent spelling removes exactly
    the entry created by another spelling of the same call."""

    @alru_cache(normalize_key=True)
    async def coro(a: int, b: int = 2, **kw: int) -> tuple[int, int, int]:
        return (a, b, sum(kw.values()))

    await coro(1)  # entry A
    await coro(1, 3)  # entry B
    await coro(2)  # entry C
    await coro(1, x=5)  # entry D
    check_lru(coro, hits=0, misses=4, cache=4, tasks=0)

    # Keyword spelling removes exactly entry A.
    assert coro.cache_invalidate(a=1, b=2)
    check_lru(coro, hits=0, misses=4, cache=3, tasks=0)
    assert not coro.cache_contains(1)
    assert coro.cache_contains(1, 3)
    assert coro.cache_contains(2)
    assert coro.cache_contains(1, x=5)

    # Default-filled spelling with **kwargs removes exactly entry D.
    assert coro.cache_invalidate(1, 2, **{"x": 5})
    check_lru(coro, hits=0, misses=4, cache=2, tasks=0)
    assert not coro.cache_contains(1, x=5)
    assert coro.cache_contains(1, 3)
    assert coro.cache_contains(2)

    # Already-removed entries are not invalidated again.
    assert not coro.cache_invalidate(1)
    assert not coro.cache_invalidate(1, x=5)
    check_lru(coro, hits=0, misses=4, cache=2, tasks=0)

    # A removed entry is recomputed on the next call, others still hit.
    assert await coro(1, b=2) == (1, 2, 0)
    assert await coro(1, 3) == (1, 3, 0)
    check_lru(coro, hits=1, misses=5, cache=3, tasks=0)


async def test_normalize_key_cache_contains_uses_same_rule(
    check_lru: Callable[..., None],
) -> None:
    @alru_cache(normalize_key=True)
    async def coro(a: int, b: int = 2) -> int:
        return a + b

    await coro(1)
    assert coro.cache_contains(1)
    assert coro.cache_contains(1, 2)
    assert coro.cache_contains(a=1, b=2)
    assert not coro.cache_contains(1, 3)
    # Lookups do not affect hit/miss counters.
    check_lru(coro, hits=0, misses=1, cache=1, tasks=0)
