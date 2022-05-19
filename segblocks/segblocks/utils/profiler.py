import time
import torch

from contextlib import contextmanager
from collections import defaultdict

class Profiler():
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.count = 0
        self.reset()
    
    def add_count(self, count=1):
        if self.enabled:
            self.count += count

    def reset(self):
        self.records = defaultdict(float)
        self.starts  = defaultdict(float)
        self.counts  = defaultdict(int)
        self.count = 0

    # start/stop
    def start(self, name):
        if self.enabled:
            torch.cuda.synchronize()
            self.starts[name] = time.perf_counter()
            self.counts[name] += 1

    def stop(self, name):
        if self.enabled:
            if name in self.starts:
                torch.cuda.synchronize()
                self.records[name] += time.perf_counter() - self.starts[name]

    def __repr__(self):
        s = ''
        if self.count > 0:
            s += f"### Profiler (images: {self.count})###\n"
            for name in sorted(self.records):
                val = self.records[name]*1000
                val_avg = val/self.count
                val_per_run = val/self.counts[name]
                
                s += f'# {name:20}: {val_avg:4.3f} ms per image (number of calls: {self.counts[name]}, per call: {val_per_run:4.3f} ms) \n'
        elif self.count == 0:
            s = '## Profiler: no batches registered'
        else:
            s = '## Profiler: disabled'
        return s
    
    @contextmanager
    def env(self, name):
        self.start(name)
        yield
        self.stop(name)

timings = Profiler(enabled=False)
