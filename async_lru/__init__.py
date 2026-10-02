import asyncio
import dataclasses
import enum
import inspect
import random
import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable, Coroutine, Hashable
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
    "ALRU_QUEUE_FULL",
    "AlruCacheLoopResetWarning",
    "AlruQueueFull",
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
class AlruQueueFull(enum.Enum):
    """Sentinel returned when no computation or queue slot is available."""

    token = enum.auto()


ALRU_QUEUE_FULL = AlruQueueFull.token


@final
class _CacheParameters(TypedDict):
    typed: bool
    maxsize: int | None
    inflight_limit: int | None
    queue_limit: int | None
    tasks: int
    closed: bool


@final
@dataclasses.dataclass(slots=True)
class _CacheItem(Generic[_R]):
    task: "asyncio.Task[_R]"
    later_call: asyncio.Handle | None
    waiters: int

    def cancel(self) -> None:
        if self.later_call is not None:
            self.later_call.cancel()
            self.later_call = None


@final
@dataclasses.dataclass(slots=True)
class _QueuedCall(Generic[_R]):
    ready: asyncio.Future[None]
    waiters: int
    fn_args: tuple[Any, ...]
    fn_kwargs: dict[str, Any]
    cache_item: _CacheItem[_R] | None = None


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
        "__inflight",
        "__queue",
        "__inflight_limit",
        "__queue_limit",
        "__closed",
        "__hits",
        "__misses",
        "__first_loop",
        "__warned_loop_reset",
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
        inflight_limit: int | None = None,
        queue_limit: int | None = None,
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
        self.__inflight: OrderedDict[Hashable, _CacheItem[_R]] = OrderedDict()
        self.__queue: OrderedDict[Hashable, _QueuedCall[_R]] = OrderedDict()
        self.__inflight_limit = inflight_limit
        self.__queue_limit = queue_limit
        self.__closed = False
        self.__hits = 0
        self.__misses = 0
        self.__first_loop: asyncio.AbstractEventLoop | None = None
        self.__warned_loop_reset = False

    @property
    def __tasks(self) -> list["asyncio.Task[_R]"]:
        return [
            cache_item.task
            for cache_item in self.__inflight.values()
            if not cache_item.task.done()
        ]

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
            self._abandon_work_for_new_loop()
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

        queued_call = self.__queue.pop(key, None)
        if queued_call is not None:
            queued_call.ready.cancel()
            return True

        cache_item = self.__cache.pop(key, None)
        if cache_item is None:
            return False
        else:
            cache_item.cancel()
            return True

    def cache_clear(self) -> None:
        self.__hits = 0
        self.__misses = 0
        for cache_item in self.__cache.values():
            if cache_item.later_call is not None:
                cache_item.later_call.cancel()
        self.__cache.clear()

    def _abandon_work_for_new_loop(self) -> None:
        self.cache_clear()
        self.__inflight = OrderedDict()
        self.__queue = OrderedDict()

    def _cancel_all_work(self) -> None:
        for queued_call in self.__queue.values():
            queued_call.ready.cancel()
        self.__queue.clear()

        for cache_item in self.__inflight.values():
            cache_item.task.cancel()

    async def cache_close(self, *, wait: bool = False) -> None:
        self.__closed = True

        if wait:
            while self.__inflight or self.__queue:
                tasks = self.__tasks
                if not tasks:
                    break
                await asyncio.gather(*tasks, return_exceptions=True)
            return

        tasks = self.__tasks
        self._cancel_all_work()

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
            inflight_limit=self.__inflight_limit,
            queue_limit=self.__queue_limit,
            tasks=len(self.__tasks),
            closed=self.__closed,
        )

    def _cache_hit(self, key: Hashable) -> None:
        self.__hits += 1
        self.__cache.move_to_end(key)

    def _cache_miss(self, key: Hashable) -> None:
        self.__misses += 1

    def _task_done_callback(self, key: Hashable, task: "asyncio.Task[_R]") -> None:
        cache_item = self.__inflight.pop(key, None)
        if cache_item is None:
            cache_item = self.__cache.get(key)
            if cache_item is None or cache_item.task is not task:
                return
            if task.cancelled() or task._exception is not None:
                self.__cache.pop(key, None)
            return

        # We must use the private attribute instead of `exception()`
        # so asyncio does not set `task.__log_traceback = False` on
        # the false assumption that the caller read the task Exception
        if task.cancelled() or task._exception is not None:
            self.__cache.pop(key, None)
        else:
            cache_item = self.__cache.get(key)
            if self.__ttl is not None and cache_item is not None:
                effective_ttl = self.__ttl
                if self.__jitter is not None:
                    effective_ttl += random.uniform(0, self.__jitter)
                loop = asyncio.get_running_loop()
                cache_item.later_call = loop.call_later(
                    effective_ttl, self.__cache.pop, key, None
                )

        self._promote_queued_calls()

    def _start_work(
        self,
        key: Hashable,
        fn_args: tuple[Any, ...],
        fn_kwargs: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
    ) -> _CacheItem[_R]:
        coro = self.__wrapped__(*fn_args, **fn_kwargs)
        task: asyncio.Task[_R] = loop.create_task(coro)
        task.add_done_callback(partial(self._task_done_callback, key))

        cache_item = _CacheItem(task, None, 1)
        self.__inflight[key] = cache_item
        self.__cache[key] = cache_item

        if self.__maxsize is not None and len(self.__cache) > self.__maxsize:
            _, dropped_cache_item = self.__cache.popitem(last=False)
            dropped_cache_item.cancel()

        return cache_item

    def _promote_queued_calls(self) -> None:
        if self.__inflight_limit is None:
            return

        loop = asyncio.get_running_loop()
        while len(self.__inflight) < self.__inflight_limit and self.__queue:
            key, queued_call = self.__queue.popitem(last=False)
            if queued_call.ready.cancelled():
                continue

            try:
                cache_item = self._start_work(
                    key,
                    queued_call.fn_args,
                    queued_call.fn_kwargs,
                    loop,
                )
            except BaseException as exc:
                if not queued_call.ready.done():
                    queued_call.ready.set_exception(exc)
                continue
            cache_item.waiters = queued_call.waiters
            queued_call.cache_item = cache_item
            queued_call.ready.set_result(None)

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
                cache_item.cancel()  # Cancel TTL expiration
                task.cancel()  # Cancel the running coroutine
                self.__cache.pop(key, None)  # Remove from cache
            raise
        finally:
            # Each logical waiter decrements waiters on exit (normal or cancelled).
            cache_item.waiters -= 1

    async def _wait_for_queued_call(
        self, key: Hashable, queued_call: _QueuedCall[_R]
    ) -> _R:
        active_item: _CacheItem[_R] | None = None
        try:
            await asyncio.shield(queued_call.ready)
            active_item = queued_call.cache_item
            if active_item is None:
                raise RuntimeError("Queued cache call was started without a task")
            return await asyncio.shield(active_item.task)
        except asyncio.CancelledError:
            active_item = active_item or queued_call.cache_item
            if active_item is None:
                if queued_call.waiters == 1:
                    self.__queue.pop(key, None)
                    queued_call.ready.cancel()
            elif active_item.waiters == 1 and not active_item.task.done():
                active_item.cancel()
                active_item.task.cancel()
                self.__cache.pop(key, None)
            raise
        finally:
            active_item = active_item or queued_call.cache_item
            if active_item is None:
                queued_call.waiters -= 1
            else:
                active_item.waiters -= 1

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
                # Each logical waiter increments waiters on entry.
                cache_item.waiters += 1
                return await self._shield_and_handle_cancelled_error(cache_item, key)

            # If the task is already done, just return the result.
            return cache_item.task.result()

        inflight_item = self.__inflight.get(key)
        if inflight_item is not None:
            self.__hits += 1
            inflight_item.waiters += 1
            return await self._shield_and_handle_cancelled_error(inflight_item, key)

        queued_call = self.__queue.get(key)
        if queued_call is not None:
            self.__hits += 1
            queued_call.waiters += 1
            return await self._wait_for_queued_call(key, queued_call)

        if (
            self.__inflight_limit is not None
            and len(self.__inflight) >= self.__inflight_limit
        ):
            if (
                self.__queue_limit is not None
                and len(self.__queue) >= self.__queue_limit
            ):
                self._cache_miss(key)
                return cast(_R, ALRU_QUEUE_FULL)

            self._cache_miss(key)
            queued_call = _QueuedCall(
                ready=loop.create_future(),
                waiters=1,
                fn_args=fn_args,
                fn_kwargs=fn_kwargs,
            )
            self.__queue[key] = queued_call
            return await self._wait_for_queued_call(key, queued_call)

        self._cache_miss(key)
        cache_item = self._start_work(key, fn_args, fn_kwargs, loop)
        return await self._shield_and_handle_cancelled_error(cache_item, key)

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
    inflight_limit: int | None = None,
    queue_limit: int | None = None,
) -> Callable[[_CBP[_R]], _LRUCacheWrapper[_R]]:
    if jitter is not None and ttl is None:
        raise ValueError("jitter requires ttl to be set")
    if jitter is not None and jitter < 0:
        raise ValueError("jitter must be non-negative")
    if key is not None and not callable(key):
        raise TypeError("key must be callable")
    if key is not None and typed:
        raise ValueError("typed is ignored when a custom key callable is provided")
    if inflight_limit is not None and (
        not isinstance(inflight_limit, int)
        or isinstance(inflight_limit, bool)
        or inflight_limit < 1
    ):
        raise ValueError("inflight_limit must be a positive integer")
    if queue_limit is not None and inflight_limit is None:
        raise ValueError("queue_limit requires inflight_limit to be set")
    if queue_limit is not None and (
        not isinstance(queue_limit, int)
        or isinstance(queue_limit, bool)
        or queue_limit < 0
    ):
        raise ValueError("queue_limit must be a non-negative integer")

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
            inflight_limit,
            queue_limit,
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
    inflight_limit: int | None = None,
    queue_limit: int | None = None,
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
    inflight_limit: int | None = None,
    queue_limit: int | None = None,
) -> Callable[[_CBP[_R]], _LRUCacheWrapper[_R]] | _LRUCacheWrapper[_R]:
    if maxsize is None or isinstance(maxsize, int):
        return _make_wrapper(
            maxsize,
            typed,
            ttl,
            jitter,
            key,
            inflight_limit,
            queue_limit,
        )
    else:
        fn = cast(_CB[_R], maxsize)

        if callable(fn) or hasattr(fn, "_make_unbound_method"):
            return _make_wrapper(128, False, None, None)(fn)

        raise NotImplementedError(f"{fn!r} decorating is not supported")
