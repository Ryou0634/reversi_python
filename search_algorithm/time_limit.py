import time


def time_limit(func):
    def quit_when_time_over(self, *args, **kwargs):
        if self.max_time and time.time() - self._start_time > self.max_time:
            raise TimeoutError
        return func(self, *args, **kwargs)

    return quit_when_time_over
