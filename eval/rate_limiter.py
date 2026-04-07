import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from typing import Callable, TypeVar


F = TypeVar("F", bound=Callable)


class RateLimiter(AbstractContextManager):
    def __init__(self, max_calls_per_minute: int = 9):
        self.max_calls = max_calls_per_minute
        self.window_seconds = 60.0
        self.timestamps: deque[float] = deque(maxlen=max_calls_per_minute)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self.timestamps and now - self.timestamps[0] >= self.window_seconds:
            self.timestamps.popleft()

    def wait_if_needed(self) -> float:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            sleep_for = 0.0
            if len(self.timestamps) >= self.max_calls:
                sleep_for = max(0.0, self.window_seconds - (now - self.timestamps[0]))
                if sleep_for > 0:
                    print(f"[RateLimiter] Sleeping {sleep_for:.2f}s to stay under {self.max_calls}/min")
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
                self._prune(now)
            self.timestamps.append(now)
            return sleep_for

    def __call__(self, func: F) -> F:
        def wrapped(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    def __enter__(self):
        self.wait_if_needed()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


global_gemini_limiter = RateLimiter(max_calls_per_minute=9)


if __name__ == "__main__":
    limiter = RateLimiter(max_calls_per_minute=9)
    call_times: list[float] = []
    start = time.monotonic()
    print("Simulating 15 calls...")
    for i in range(15):
        slept = limiter.wait_if_needed()
        t = time.monotonic() - start
        call_times.append(t)
        print(f"Call {i+1:02d} at +{t:6.2f}s (slept {slept:5.2f}s)")

    total = call_times[-1] if call_times else 0.0
    print(f"Total elapsed: {total:.2f}s")

    max_in_window = 0
    for idx, ts in enumerate(call_times):
        cnt = sum(1 for t in call_times if ts - 60.0 < t <= ts)
        max_in_window = max(max_in_window, cnt)
        if idx == len(call_times) - 1:
            break
    assert total >= 60.0, "Expected total elapsed to be at least 60s for 15 calls at 9 RPM"
    assert max_in_window <= 9, f"More than 9 calls in a 60s window: {max_in_window}"
    print("Rate limiter self-test passed.")

