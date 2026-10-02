import asyncio
import contextvars
import dataclasses
import inspect
import random
import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Hashable, Iterable, Sequence
from functools import _CacheInfo, _make_key, partial, partialmethod
from typing import Any, Generic, TypedDict, TypeVar, cast, final, overload


if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info < (3, 14):
    from asyncio.coroutines import _is_coroutine  # type: ignore[attr-defined]


__version__ = "2.3.0"

__all__ = (
    "AlruCacheCall",
    "AlruCacheLoopResetWarning",
    "AlruRecursionError",
    "AlruRecursiveWaitError",
    "alru_cache",
)


_T = TypeVar("_T")
_R = TypeVar("_R")
_Coro = Coroutine[Any, Any, _R]
_CB = Callable[..., _Coro[_R]]
_CBP = _CB[_R] | partial[_Coro[_R]] | partialmethod[_Coro[_R]]


class AlruCacheLoopResetWarning(UserWarning):
    """Emitted once per cache instance when a loop change triggers an auto-reset."""


@final
@dataclasses.dataclass(frozen=True, slots=True)
class AlruCacheCall:
    """A cached function call in a recursive wait diagnostic chain."""

    function: str
    key: Hashable

    def __repr__(self) -> str:
        return (
            f"AlruCacheCall(function={self.function!r}, "
            f"key={_key_repr(self.key)})"
        )


class AlruRecursiveWaitError(RecursionError):
    """Raised when a cached computation waits for itself in one causal chain."""

    def __init__(
        self, calls: Iterable[tuple["_LRUCacheWrapper[Any]", Hashable]]
    ) -> None:
        self.chain: tuple[AlruCacheCall, ...] = tuple(
            AlruCacheCall(_function_label(wrapper), key) for wrapper, key in calls
        )
        chain = " -> ".join(
            f"{call.function}({_key_repr(call.key)})" for call in self.chain
        )
        super().__init__(f"Recursive cached call wait detected: {chain}")


AlruRecursionError = AlruRecursiveWaitError


def _key_repr(key: Hashable, depth: int = 0) -> str:
    try:
        key_type = type(key)
    except Exception:
        return "<unrepresentable key>"
    if depth >= 3:
        return "..."
    type_name = _type_name(key_type)

    if key_type is str:
        text_key = cast(str, key)
        value = (
            text_key
            if str.__len__(text_key) <= 70
            else str.__getitem__(text_key, slice(70)) + "..."
        )
        return str.__repr__(value)
    if key_type is bytes:
        bytes_key = cast(bytes, key)
        value = (
            bytes_key
            if bytes.__len__(bytes_key) <= 65
            else bytes.__getitem__(bytes_key, slice(65)) + b"..."
        )
        return bytes.__repr__(value)
    if key_type is bool:
        return bool.__repr__(cast(bool, key))
    if key_type is int:
        int_key = cast(int, key)
        if int.bit_length(int_key) > 256:
            return f"<large {type_name} key>"
        text = int.__repr__(int_key)
        return text if len(text) <= 80 else text[:77] + "..."
    if key_type is float:
        return float.__repr__(cast(float, key))
    if key is None:
        return "None"
    if key_type is type:
        return f"<class {_type_name(cast(type[Any], key))}>"
    if key_type is tuple:
        return _bounded_sequence_repr(
            cast(Sequence[Hashable], key), "(", ")", depth, allow_comma=True
        )
    if (
        type_name == "_HashedSeq"
        and getattr(key_type, "__module__", "") == "functools"
    ):
        return _bounded_sequence_repr(
            cast(Sequence[Hashable], key),
            "functools._HashedSeq[",
            "]",
            depth,
        )
    return f"<unrepresentable {type_name} key>"


def _short_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        text = str.__str__(value)
        if str.__len__(text) <= limit:
            return text
        return str.__getitem__(text, slice(limit - 3)) + "..."
    except Exception:
        return None


def _type_name(key_type: type[Any]) -> str:
    try:
        name = key_type.__qualname__
    except Exception:
        try:
            name = key_type.__name__
        except Exception:
            name = "unknown"
    return _short_text(name, 80) or "unknown"


def _bounded_sequence_repr(
    value: Sequence[Hashable],
    prefix: str,
    suffix: str,
    depth: int,
    *,
    allow_comma: bool = False,
) -> str:
    parts = []
    for index, item in enumerate(value):
        if index == 8:
            parts.append("...")
            break
        parts.append(_key_repr(item, depth + 1))
    if allow_comma and len(value) == 1:
        parts.append("")
    return prefix + ", ".join(parts) + suffix


def _function_label(wrapper: "_LRUCacheWrapper[Any]") -> str:
    try:
        target: Any = wrapper
        for _ in range(8):
            if isinstance(target, (partial, partialmethod)):
                target = target.func
                continue
            wrapped = getattr(target, "__wrapped__", None)
            if wrapped is not None:
                target = wrapped
                continue
            break

        qualname = getattr(target, "__qualname__", None) or getattr(
            target, "__name__", None
        )
        if qualname is None:
            qualname = _type_name(type(target))
        module = getattr(target, "__module__", None)
        if (
            module
            and module != "builtins"
            and not qualname.startswith(module + ".")
        ):
            label = f"{module}.{qualname}"
        else:
            label = qualname
        return _short_text(label, 100) or "unknown cached function"
    except Exception:
        return "unknown cached function"


@final
@dataclasses.dataclass(slots=True)
class _ActiveCacheCall:
    wrapper: "_LRUCacheWrapper[Any] | None"
    key: Hashable | None
    active: bool = True


_active_cache_calls: contextvars.ContextVar[tuple[_ActiveCacheCall, ...]] = (
    contextvars.ContextVar("_active_cache_calls", default=())
)


@final
class _CacheParameters(TypedDict):
    typed: bool
    maxsize: int | None
    tasks: int
    closed: bool


@final
@dataclasses.dataclass(slots=True)
class _CacheItem(Generic[_R]):
    task: "asyncio.Task[_R]"
    later_call: asyncio.Handle | None
    waiters: int
    active_call: _ActiveCacheCall | None = None

    def cancel(self) -> None:
        if self.later_call is not None:
            self.later_call.cancel()

    def deactivate(self) -> None:
        self.cancel()
        self.finish_active_call()

    def finish_active_call(self) -> None:
        if self.active_call is not None:
            self.active_call.active = False
            self.active_call.wrapper = None
            self.active_call.key = None
            self.active_call = None


@final
class _LRUCacheWrapper(Generic[_R]):
    __slots__ = (
        "__wrapped__",
        "__maxsize",
        "__typed",
        "__ttl",
        "__jitter",
        "__key",
        "__cache",
        "__closed",
        "__hits",
        "__misses",
        "__first_loop",
        "__warned_loop_reset",
        "__recursive_wait_detection",
        "_is_coroutine",
        "_is_coroutine_marker",
        "__dict__",
    )

    def __init__(
        self,
        fn: _CB[_R],
        maxsize: int | None,
        typed: bool,
        ttl: float | None,
        jitter: float | None,
        key: Callable[..., Hashable] | None = None,
        recursive_wait_detection: bool = False,
    ) -> None:
        try:
            self.__module__ = fn.__module__
        except AttributeError:
            pass
        try:
            self.__name__ = fn.__name__
        except AttributeError:
            pass
        try:
            self.__qualname__ = fn.__qualname__
        except AttributeError:
            pass
        try:
            self.__doc__ = fn.__doc__
        except AttributeError:
            pass
        # Copy the lazy __annotate__ function; reading fn.__annotations__
        # would force evaluation of deferred annotations (PEP 649).
        if sys.version_info >= (3, 14):
            try:
                self.__annotate__ = fn.__annotate__
            except AttributeError:
                try:
                    self.__annotations__ = fn.__annotations__
                except AttributeError:
                    pass
        else:
            try:
                self.__annotations__ = fn.__annotations__
            except AttributeError:
                pass
        try:
            self.__dict__.update(fn.__dict__)
        except AttributeError:
            pass
        # set __wrapped__ last so we don't inadvertently copy it
        # from the wrapped function when updating __dict__
        if sys.version_info < (3, 14):
            self._is_coroutine = _is_coroutine
        self.__wrapped__ = fn
        self.__maxsize = maxsize
        self.__typed = typed
        self.__ttl = ttl
        self.__jitter = jitter
        self.__key = key
        self.__cache: OrderedDict[Hashable, _CacheItem[_R]] = OrderedDict()
        self.__closed = False
        self.__hits = 0
        self.__misses = 0
        self.__first_loop: asyncio.AbstractEventLoop | None = None
        self.__warned_loop_reset = False
        self.__recursive_wait_detection = recursive_wait_detection

    @property
    def __tasks(self) -> list["asyncio.Task[_R]"]:
        # NOTE: I don't think we need to form a set first here but not
        # too sure we want it for guarantees
        return list(
            {
                cache_item.task
                for cache_item in self.__cache.values()
                if not cache_item.task.done()
            }
        )

    def _check_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if self.__first_loop is None:
            self.__first_loop = loop
        elif self.__first_loop is not loop:
            if not self.__warned_loop_reset:
                warnings.warn(
                    "alru_cache detected event loop change and auto-cleared "
                    "stale entries. This is safe but unusual outside of "
                    "tests (pytest-anyio, etc.).",
                    AlruCacheLoopResetWarning,
                    stacklevel=3,
                )
                self.__warned_loop_reset = True
            # Old cache entries hold tasks/handles bound to the previous
            # loop and are invalid here.  Clear and rebind.
            self.cache_clear()
            self.__first_loop = loop

    def cache_contains(self, /, *args: Hashable, **kwargs: Any) -> bool:
        """Check if the given arguments are in the cache.

        Does not affect hit/miss counters or LRU ordering.
        """
        # Inlined instead of a shared helper: an extra method call is
        # measurable on these short code paths (CodSpeed).
        if self.__key is not None:
            key = self.__key(*args, **kwargs)
        else:
            key = _make_key(args, kwargs, self.__typed)
        return key in self.__cache

    def cache_invalidate(self, /, *args: Hashable, **kwargs: Any) -> bool:
        if self.__key is not None:
            key = self.__key(*args, **kwargs)
        else:
            key = _make_key(args, kwargs, self.__typed)

        cache_item = self.__cache.pop(key, None)
        if cache_item is None:
            return False
        else:
            cache_item.deactivate()
            return True

    def cache_clear(self) -> None:
        self.__hits = 0
        self.__misses = 0

        for cache_item in self.__cache.values():
            cache_item.deactivate()
        self.__cache.clear()

    async def cache_close(self, *, wait: bool = False) -> None:
        self.__closed = True

        tasks = self.__tasks
        if not tasks:
            return

        if not wait:
            for task in tasks:
                if not task.done():
                    task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

    def cache_info(self) -> _CacheInfo:
        return _CacheInfo(
            self.__hits,
            self.__misses,
            self.__maxsize,
            len(self.__cache),
        )

    def cache_parameters(self) -> _CacheParameters:
        return _CacheParameters(
            maxsize=self.__maxsize,
            typed=self.__typed,
            tasks=len(self.__tasks),
            closed=self.__closed,
        )

    def _cache_hit(self, key: Hashable) -> None:
        self.__hits += 1
        self.__cache.move_to_end(key)

    def _cache_miss(self, key: Hashable) -> None:
        self.__misses += 1

    def _task_done_callback(self, key: Hashable, task: "asyncio.Task[_R]") -> None:
        # We must use the private attribute instead of `exception()`
        # so asyncio does not set `task.__log_traceback = False` on
        # the false assumption that the caller read the task Exception
        if task.cancelled() or task._exception is not None:
            cache_item = self.__cache.pop(key, None)
            if cache_item is not None:
                cache_item.deactivate()
            return

        cache_item = self.__cache.get(key)
        if cache_item is not None:
            if self.__ttl is not None:
                effective_ttl = self.__ttl
                if self.__jitter is not None:
                    effective_ttl += random.uniform(0, self.__jitter)
                loop = asyncio.get_running_loop()
                cache_item.later_call = loop.call_later(
                    effective_ttl, self._expire_cache_item, key
                )
            cache_item.finish_active_call()

    def _expire_cache_item(self, key: Hashable) -> None:
        cache_item = self.__cache.pop(key, None)
        if cache_item is not None:
            cache_item.deactivate()

    def _recursive_wait(
        self, cache_item: _CacheItem[Any]
    ) -> tuple[int, _ActiveCacheCall] | None:
        active_call = cache_item.active_call
        if active_call is None or not active_call.active:
            return None

        chain = _active_cache_calls.get()
        for index, current_call in enumerate(chain):
            if (
                current_call is active_call
                and current_call.active
                and current_call.wrapper is not None
                and current_call.key is not None
            ):
                return index, active_call
        return None

    async def _shield_and_handle_cancelled_error(
        self, cache_item: _CacheItem[_T], key: Hashable
    ) -> _T:
        task = cache_item.task
        try:
            # All waiters await the same shielded task.
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # If this is the last waiter and the underlying task is not done,
            # cancel the underlying task and remove the cache entry.
            if cache_item.waiters == 1 and not task.done():
                cache_item.deactivate()  # Cancel TTL expiration
                task.cancel()  # Cancel the running coroutine
                self.__cache.pop(key, None)  # Remove from cache
            raise
        finally:
            # Each logical waiter decrements waiters on exit (normal or cancelled).
            cache_item.waiters -= 1

    async def __call__(self, /, *fn_args: Any, **fn_kwargs: Any) -> _R:
        if self.__closed:
            raise RuntimeError(f"alru_cache is closed for {self}")

        loop = asyncio.get_running_loop()
        self._check_loop(loop)

        if self.__key is not None:
            key = self.__key(*fn_args, **fn_kwargs)
        else:
            key = _make_key(fn_args, fn_kwargs, self.__typed)
        cache_item = self.__cache.get(key)

        if cache_item is not None:
            self._cache_hit(key)
            if not cache_item.task.done():
                if self.__recursive_wait_detection or _active_cache_calls.get():
                    recursive_wait = self._recursive_wait(cache_item)
                    if recursive_wait is not None:
                        recursive_index, target_call = recursive_wait
                        diagnostic_chain = tuple(
                            (current_call.wrapper, current_call.key)
                            for current_call in _active_cache_calls.get()
                            if current_call.wrapper is not None
                            and current_call.key is not None
                        )
                        target = (target_call.wrapper, target_call.key)
                        raise AlruRecursiveWaitError(
                            (*diagnostic_chain[recursive_index:], target)
                        )

                # Each logical waiter increments waiters on entry.
                cache_item.waiters += 1
                return await self._shield_and_handle_cancelled_error(cache_item, key)

            # If the task is already done, just return the result.
            return cache_item.task.result()

        coro = self.__wrapped__(*fn_args, **fn_kwargs)
        active_call: _ActiveCacheCall | None = None
        cache_item: _CacheItem[_R] | None = None
        token: (
            contextvars.Token[tuple[_ActiveCacheCall, ...]] | None
        ) = None

        if self.__recursive_wait_detection or _active_cache_calls.get():
            active_call = _ActiveCacheCall(self, key)
            token = _active_cache_calls.set(
                _active_cache_calls.get() + (active_call,)
            )

        try:
            task: asyncio.Task[_R] = loop.create_task(coro)
            task.add_done_callback(partial(self._task_done_callback, key))

            cache_item = _CacheItem(task, None, 1, active_call)
            self.__cache[key] = cache_item

            if self.__maxsize is not None and len(self.__cache) > self.__maxsize:
                _, dropped_cache_item = self.__cache.popitem(last=False)
                dropped_cache_item.deactivate()

            self._cache_miss(key)

            return await self._shield_and_handle_cancelled_error(cache_item, key)
        finally:
            if active_call is not None and cache_item is None:
                active_call.active = False
            if token is not None:
                _active_cache_calls.reset(token)

    def __get__(
        self, instance: _T, owner: type[_T] | None
    ) -> Self | "_LRUCacheWrapperInstanceMethod[_R, _T]":
        if owner is None:
            return self
        else:
            return _LRUCacheWrapperInstanceMethod(self, instance)


@final
class _LRUCacheWrapperInstanceMethod(Generic[_R, _T]):
    __slots__ = (
        "__dict__",
        "_is_coroutine",
        "__wrapped__",
        "__instance",
        "__wrapper",
    )

    def __init__(
        self,
        wrapper: _LRUCacheWrapper[_R],
        instance: _T,
    ) -> None:
        try:
            self.__module__ = wrapper.__module__
        except AttributeError:
            pass
        try:
            self.__name__ = wrapper.__name__
        except AttributeError:
            pass
        try:
            self.__qualname__ = wrapper.__qualname__
        except AttributeError:
            pass
        try:
            self.__doc__ = wrapper.__doc__
        except AttributeError:
            pass
        # Prefer the lazy __annotate__ function, see the matching logic
        # in _LRUCacheWrapper.__init__.
        if sys.version_info >= (3, 14):
            try:
                self.__annotate__ = wrapper.__annotate__
            except AttributeError:
                try:
                    self.__annotations__ = wrapper.__annotations__
                except AttributeError:
                    pass
        else:
            try:
                self.__annotations__ = wrapper.__annotations__
            except AttributeError:
                pass
        try:
            self.__dict__.update(wrapper.__dict__)
        except AttributeError:
            pass
        # set __wrapped__ last so we don't inadvertently copy it
        # from the wrapped function when updating __dict__
        if sys.version_info < (3, 14):
            self._is_coroutine = _is_coroutine
        self.__wrapped__ = wrapper.__wrapped__
        self.__instance = instance
        self.__wrapper = wrapper

    def cache_contains(self, /, *args: Hashable, **kwargs: Any) -> bool:
        return self.__wrapper.cache_contains(self.__instance, *args, **kwargs)

    def cache_invalidate(self, /, *args: Hashable, **kwargs: Any) -> bool:
        return self.__wrapper.cache_invalidate(self.__instance, *args, **kwargs)

    def cache_clear(self) -> None:
        self.__wrapper.cache_clear()

    async def cache_close(
        self,
        *,
        wait: bool = False,
        cancel: bool = False,
        return_exceptions: bool = True,
    ) -> None:
        if cancel or return_exceptions is not True:
            warnings.warn(
                "cancel/return_exceptions are deprecated; use wait=True to allow tasks "
                "to finish and wait=False to cancel pending tasks.",
                DeprecationWarning,
                stacklevel=2,
            )
        await self.__wrapper.cache_close(wait=wait)

    def cache_info(self) -> _CacheInfo:
        return self.__wrapper.cache_info()

    def cache_parameters(self) -> _CacheParameters:
        return self.__wrapper.cache_parameters()

    async def __call__(self, /, *fn_args: Any, **fn_kwargs: Any) -> _R:
        return await self.__wrapper(self.__instance, *fn_args, **fn_kwargs)


def _make_wrapper(
    maxsize: int | None,
    typed: bool,
    ttl: float | None = None,
    jitter: float | None = None,
    key: Callable[..., Hashable] | None = None,
    recursive_wait_detection: bool = False,
) -> Callable[[_CBP[_R]], _LRUCacheWrapper[_R]]:
    if jitter is not None and ttl is None:
        raise ValueError("jitter requires ttl to be set")
    if jitter is not None and jitter < 0:
        raise ValueError("jitter must be non-negative")
    if key is not None and not callable(key):
        raise TypeError("key must be callable")
    if key is not None and typed:
        raise ValueError("typed is ignored when a custom key callable is provided")

    def wrapper(fn: _CBP[_R]) -> _LRUCacheWrapper[_R]:
        origin = fn

        while isinstance(origin, (partial, partialmethod)):
            origin = origin.func

        if not inspect.iscoroutinefunction(origin):
            raise RuntimeError(f"Coroutine function is required, got {fn!r}")

        if hasattr(fn, "_make_unbound_method"):
            fn = fn._make_unbound_method()

        wrapper = _LRUCacheWrapper(
            cast(_CB[_R], fn),
            maxsize,
            typed,
            ttl,
            jitter,
            key,
            recursive_wait_detection,
        )
        if sys.version_info >= (3, 12):
            wrapper = inspect.markcoroutinefunction(wrapper)
        return wrapper

    return wrapper


@overload
def alru_cache(
    maxsize: int | None = 128,
    typed: bool = False,
    *,
    ttl: float | None = None,
    jitter: float | None = None,
    key: Callable[..., Hashable] | None = None,
    recursive_wait_detection: bool = False,
    detect_recursion: bool | None = None,
) -> Callable[[_CBP[_R]], _LRUCacheWrapper[_R]]:
    ...


@overload
def alru_cache(
    maxsize: _CBP[_R],
    /,
) -> _LRUCacheWrapper[_R]:
    ...


def alru_cache(
    maxsize: int | _CBP[_R] | None = 128,
    typed: bool = False,
    *,
    ttl: float | None = None,
    jitter: float | None = None,
    key: Callable[..., Hashable] | None = None,
    recursive_wait_detection: bool = False,
    detect_recursion: bool | None = None,
) -> Callable[[_CBP[_R]], _LRUCacheWrapper[_R]] | _LRUCacheWrapper[_R]:
    if detect_recursion is not None:
        if recursive_wait_detection:
            raise ValueError(
                "detect_recursion is an alias for recursive_wait_detection; "
                "do not pass both"
            )
        recursive_wait_detection = detect_recursion

    if maxsize is None or isinstance(maxsize, int):
        return _make_wrapper(
            maxsize,
            typed,
            ttl,
            jitter,
            key,
            recursive_wait_detection,
        )
    else:
        fn = cast(_CB[_R], maxsize)

        if callable(fn) or hasattr(fn, "_make_unbound_method"):
            if recursive_wait_detection or detect_recursion is not None:
                raise ValueError("recursive wait detection requires a called decorator")
            return _make_wrapper(128, False, None, None)(fn)

        raise NotImplementedError(f"{fn!r} decorating is not supported")
