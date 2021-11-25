import time


def set_start_time(func):
    def _set_start_time(self, *args, **kwargs):
        self._start_time = time.time()
        return func(self, *args, **kwargs)

    return _set_start_time


def quit_when_time_over(func):
    def _quit_when_time_over(self, *args, **kwargs):
        if self.max_time is not None and time.time() - self._start_time > self.max_time:
            raise TimeoutError
        return func(self, *args, **kwargs)

    return _quit_when_time_over
