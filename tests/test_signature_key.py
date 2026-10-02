import asyncio
from collections.abc import Callable
from functools import partial, partialmethod

import pytest

from async_lru import _CacheParameters, alru_cache


async def test_signature_key_equivalent_spellings(
    check_lru: Callable[..., None]
) -> None:
    """Different spellings of the same bound call share one entry."""

    @alru_cache(signature_key=True)
    async def coro(
        x: int, mode: str = "a", **extra: object
    ) -> tuple[object, ...]:
        return (x, mode, extra)

    equivalent_calls = [
        lambda: coro(1),
        lambda: coro(x=1),
        lambda: coro(1, "a"),
        lambda: coro(1, mode="a"),
        lambda: coro(x=1, mode="a"),
        lambda: coro(mode="a", x=1),
    ]

    for call in equivalent_calls:
        assert await call() == (1, "a", {})

    check_lru(coro, hits=5, misses=1, cache=1, tasks=0)
    assert coro.cache_contains(1)
    assert coro.cache_contains(x=1)
    assert coro.cache_contains(1, "a")
    assert coro.cache_contains(mode="a", x=1)


async def test_signature_key_non_equivalent_groups(
    check_lru: Callable[..., None]
) -> None:

    @alru_cache(signature_key=True)
    async def coro(
        x: int, mode: str = "a", **extra: int
    ) -> tuple[object, ...]:
        return (x, mode, extra)

    await coro(1)
    await coro(2)
    await coro(1, mode="b")
    await coro(1, n=1)
    await coro(x=1, n=2)
    check_lru(coro, hits=0, misses=5, cache=5, tasks=0)

    # Different spellings of these same calls are still hits.
    assert await coro(mode="b", x=1) == (1, "b", {})
    assert await coro(1, n=2) == (1, "a", {"n": 2})
    check_lru(coro, hits=2, misses=5, cache=5, tasks=0)


async def test_signature_key_positional_and_keyword_only(
    check_lru: Callable[..., None]
) -> None:

    @alru_cache(signature_key=True)
    async def coro(a: int, /, b: int, *, k: int = 3) -> tuple[int, int, int]:
        return (a, b, k)

    await coro(1, 2)
    await coro(1, b=2)
    await coro(1, 2, k=3)
    await coro(1, 2)
    check_lru(coro, hits=3, misses=1, cache=1, tasks=0)

    await coro(1, 2, k=4)
    check_lru(coro, hits=3, misses=2, cache=2, tasks=0)


async def test_signature_key_varargs_and_varkwargs(
    check_lru: Callable[..., None]
) -> None:

    @alru_cache(signature_key=True)
    async def coro(
        a: int, *nums: int, k: int = 0, **extra: int
    ) -> tuple[object, ...]:
        return (a, nums, k, extra)

    await coro(1, 2, 3)
    await coro(1, 2, 3, k=0)
    await coro(a=1, k=0)
    check_lru(coro, hits=1, misses=2, cache=2, tasks=0)

    # **kwargs spelling order is canonicalized.
    before = coro.cache_info().currsize
    await coro(1, x=1, y=2)
    await coro(1, y=2, x=1)
    await coro(a=1, y=2, x=1)
    assert coro.cache_info().currsize == before + 1

    check_lru(coro, hits=3, misses=3, cache=3, tasks=0)


async def test_signature_key_invalidate_other_spelling(
    check_lru: Callable[..., None]
) -> None:
    """Populate via one spelling, invalidate via another: exactly the
    matching cached entry disappears."""

    @alru_cache(signature_key=True)
    async def coro(x: int, mode: str = "a", **extra: int) -> tuple:
        return (x, mode, extra)

    await coro(1)
    await coro(1, mode="b")
    await coro(1, n=1)
    check_lru(coro, hits=0, misses=3, cache=3, tasks=0)

    # Invalidate the default-mode entry using a different spelling.
    assert coro.cache_invalidate(x=1, mode="a")
    check_lru(coro, hits=0, misses=3, cache=2, tasks=0)
    assert not coro.cache_contains(1)
    assert coro.cache_contains(1, mode="b")
    assert coro.cache_contains(1, n=1)

    # The other two entries are unaffected.
    assert not coro.cache_invalidate(1)
    assert coro.cache_invalidate(1, mode="b")
    check_lru(coro, hits=0, misses=3, cache=1, tasks=0)
    assert coro.cache_invalidate(1, n=1)
    check_lru(coro, hits=0, misses=3, cache=0, tasks=0)
    assert not coro.cache_invalidate(1, n=1)


async def test_signature_key_contains_invalidate_unhashable() -> None:
    @alru_cache(signature_key=True)
    async def coro(x: int, opt: object = None) -> object:
        return opt

    with pytest.raises(TypeError, match="unhashable"):
        await coro(1, [1, 2])
    assert coro.cache_info().currsize == 0

    with pytest.raises(TypeError, match="unhashable"):
        coro.cache_contains(1, [1, 2])

    with pytest.raises(TypeError, match="unhashable"):
        coro.cache_invalidate(1, [1, 2])


async def test_signature_key_mutable_default_is_not_stable_key() -> None:
    """A mutable default must not be stringified into a stable key."""

    @alru_cache(signature_key=True)
    async def coro(x: int, opt: list[int] = []) -> list[int]:  # noqa: B006
        return opt

    with pytest.raises(TypeError, match="unhashable"):
        await coro(1)
    with pytest.raises(TypeError, match="unhashable"):
        coro.cache_invalidate(1)
    with pytest.raises(TypeError, match="unhashable"):
        coro.cache_contains(1)
    assert coro.cache_info().currsize == 0

    # Explicit hashable default spellings still collapse.
    @alru_cache(signature_key=True)
    async def coro2(x: int, opt: int = 5) -> int:
        return opt

    await coro2(1)
    await coro2(1, 5)
    await coro2(x=1, opt=5)
    assert coro2.cache_info().currsize == 1


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((), {}),
        ((1,), {"x": 2}),
        ((1, 2), {"y": 3}),
        ((1,), {"unknown_kwarg": 1}),
    ],
)
async def test_signature_key_invalid_call_matches_native(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    @alru_cache(signature_key=True)
    async def coro(x: int, y: int = 1, *nums: int) -> tuple:
        return (x, y, nums)

    with pytest.raises(TypeError) as native:
        probe = coro.__wrapped__(*args, **kwargs)
        probe.close()
    expected_message = str(native.value)

    with pytest.raises(TypeError) as raised:
        await coro(*args, **kwargs)
    assert str(raised.value) == expected_message

    with pytest.raises(TypeError) as raised_contains:
        coro.cache_contains(*args, **kwargs)
    assert str(raised_contains.value) == expected_message

    with pytest.raises(TypeError) as raised_invalidate:
        coro.cache_invalidate(*args, **kwargs)
    assert str(raised_invalidate.value) == expected_message


async def test_signature_key_positional_only_kwarg_error() -> None:
    @alru_cache(signature_key=True)
    async def coro(x: int, /) -> int:
        return x

    with pytest.raises(
        TypeError, match="positional-only arguments passed as keyword"
    ):
        await coro(x=1)
    with pytest.raises(
        TypeError, match="positional-only arguments passed as keyword"
    ):
        coro.cache_invalidate(x=1)


async def test_signature_key_typed_separates_bool_int(
    check_lru: Callable[..., None]
) -> None:

    @alru_cache(signature_key=True, typed=True)
    async def coro(x: int) -> int:
        return x

    await coro(1)
    await coro(True)
    await coro(1.0)
    check_lru(coro, hits=0, misses=3, cache=3, tasks=0)

    # Cross-spelling still collapses.
    await coro(x=True)
    check_lru(coro, hits=1, misses=3, cache=3, tasks=0)


async def test_signature_key_untyped_kwargs_merge_bool_int() -> None:
    # Mirror functools semantics: without typed, keyword-spelled values
    # that compare equal merge (including True/1).
    @alru_cache(signature_key=True)
    async def coro(*, x: int) -> int:
        return x

    await coro(x=1)
    await coro(x=True)
    assert coro.cache_info().currsize == 1
    assert await coro(x=True) == 1


async def test_signature_key_bound_method(check_lru: Callable[..., None]) -> None:
    class Api:
        def __init__(self, name: str) -> None:
            self.name = name

        @alru_cache(signature_key=True)
        async def fetch(self, query: str, mode: str = "a") -> str:
            return f"{self.name}:{query}:{mode}"

    first = Api("first")
    second = Api("second")

    assert await first.fetch("q") == "first:q:a"
    assert await first.fetch("q", "a") == "first:q:a"
    assert await first.fetch(query="q", mode="a") == "first:q:a"
    check_lru(Api.fetch, hits=2, misses=1, cache=1, tasks=0)

    # Instances are still isolated: self participates in the key.
    assert await second.fetch("q") == "second:q:a"
    check_lru(Api.fetch, hits=2, misses=2, cache=2, tasks=0)

    assert first.fetch.cache_contains("q")
    assert second.fetch.cache_contains(query="q", mode="a")

    # Invalidate via another spelling: exactly the first entry goes.
    assert first.fetch.cache_invalidate(query="q", mode="a")
    check_lru(Api.fetch, hits=2, misses=2, cache=1, tasks=0)
    assert not first.fetch.cache_contains("q")
    assert second.fetch.cache_contains("q")

    # Invalid method calls carry the native method name.
    with pytest.raises(TypeError, match=r"fetch\(\) missing 1 required"):
        await first.fetch()
    with pytest.raises(TypeError, match=r"fetch\(\) missing 1 required"):
        first.fetch.cache_invalidate()


async def test_signature_key_partialmethod(check_lru: Callable[..., None]) -> None:
    class Obj:
        async def _coro(self, val: int, extra: int = 2) -> tuple[int, int]:
            return (val, extra)

        coro = alru_cache(partialmethod(_coro, 1), signature_key=True)

    obj = Obj()
    assert await obj.coro() == (1, 2)
    assert await obj.coro(extra=2) == (1, 2)
    check_lru(Obj.coro, hits=1, misses=1, cache=1, tasks=0)

    assert obj.coro.cache_invalidate()
    check_lru(Obj.coro, hits=1, misses=1, cache=0, tasks=0)


async def test_signature_key_partial(check_lru: Callable[..., None]) -> None:
    async def base(a: int, b: int, *, k: int = 3) -> tuple[int, int, int]:
        return (a, b, k)

    coro = alru_cache(partial(base, 1), signature_key=True)

    assert await coro(2) == (1, 2, 3)
    assert await coro(b=2) == (1, 2, 3)
    assert await coro(2, k=3) == (1, 2, 3)
    check_lru(coro, hits=2, misses=1, cache=1, tasks=0)


def test_signature_key_with_custom_key_raises() -> None:
    with pytest.raises(ValueError, match="signature_key"):
        alru_cache(key=lambda x: x, signature_key=True)

    async def coro(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="signature_key"):
        alru_cache(coro, key=lambda x: x, signature_key=True)


async def test_signature_key_concurrent_call_deduplicates() -> None:
    running = 0

    @alru_cache(signature_key=True)
    async def coro(x: int, mode: str = "a") -> int:
        nonlocal running
        running += 1
        assert running == 1
        await asyncio.sleep(0.01)
        running -= 1
        return x

    results = await asyncio.gather(coro(1), coro(x=1, mode="a"), coro(1, "a"))
    assert results == [1, 1, 1]
    assert coro.cache_info().currsize == 1


def test_signature_key_cache_parameters() -> None:
    @alru_cache(signature_key=True)
    async def coro(x: int) -> int:
        return x

    assert coro.cache_parameters() == _CacheParameters(
        typed=False,
        signature_key=True,
        maxsize=128,
        tasks=0,
        closed=False,
    )

    @alru_cache(typed=True, signature_key=True)
    async def coro2(x: int) -> int:
        return x

    assert coro2.cache_parameters()["typed"] is True
    assert coro2.cache_parameters()["signature_key"] is True


def test_signature_key_off_by_default_keeps_spellings_distinct() -> None:
    @alru_cache()
    async def coro(x: int, mode: str = "a") -> tuple[int, str]:
        return (x, mode)

    async def scenario() -> None:
        await coro(1)
        await coro(x=1)
        assert coro.cache_info().currsize == 2

    asyncio.run(scenario())
