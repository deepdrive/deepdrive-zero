from collections import deque

import numpy as np


class ExperienceBuffer:
    def __init__(self, step_seconds=0.25, seconds_to_keep=2,
                 fade_fn=None):
        self.step_seconds: float = step_seconds
        self.seconds_to_keep: int = seconds_to_keep

        self.max_length = int(self.seconds_to_keep / self.step_seconds)
        self.buffer = deque(maxlen=self.max_length)
        self.last_capture_time = None
        self.fade_fn = fade_fn or (lambda j: j)

        self.fade_length = 0
        while self.fade_fn(self.fade_length) < self.max_length:
            self.fade_length += 1

        self.blank_buffer = None

    def maybe_add(self, x, t):
        if self.last_capture_time is None or \
                t >= (self.last_capture_time + self.step_seconds):
            self.buffer.appendleft(x)
            self.last_capture_time = t

    def setup(self, shape: tuple = None):
        shape = shape or (1,)
        for _ in range(self.max_length):
            self.buffer.append(np.zeros(shape))
        self.blank_buffer = self.buffer

    def reset(self):
        self.buffer.clear()

    def size(self):
        return len(self.buffer)

    def __len__(self):
        return self.size()


def sanity():
    e = ExperienceBuffer()
    e.reset()
    for i in range(e.max_length):
        e.maybe_add(np.array(i), i * e.step_seconds)
    assert e.size() == e.max_length
    assert e.buffer[-1] == e.max_length - 1


if __name__ == '__main__':
    sanity()
