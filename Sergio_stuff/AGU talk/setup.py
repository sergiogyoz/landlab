# %%
# imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colorsmaps
import numpy as np
import math
import time as pytimer

# grid setup
class setup:

    def __init__(self, shape, steepness=0.1):
        self.n = shape[0]
        self.m = shape[1]
        self.steepness = steepness
        self.grid = [0] * (self.n * self.m)
        self.timer = 0.0

    def line(self, Vertical=False):
        if not Vertical:
            max = self.steepness * self.m * 1.2 + 0.01
            self.grid = [max] * (3 * self.m)
            for i in range(self.m):
                self.grid[self.m + i] = self.steepness * (self.m - i)
        else:
            max = self.steepness * self.n * 1.2 + 0.01
            self.grid = [max] * (3 * self.n)
            for i in range(self.n):
                self.grid[3 * i + 1] = self.steepness * i
        return self.grid

    @staticmethod
    def custom(id=1):
        match id:
            case 1:
                return [20.0, 20.0, 0.0, 20.0, 20.0, 20.0,
                        20.0, 11.0, 3.0, 2.0, 11.0, 20.0,
                        20.0, 12.0, 3.0, 4.0, 12.0, 20.0,
                        20.0, 13.0, 5.0, 4.0, 13.0, 20.0,
                        20.0, 14.0, 5.0, 6.0, 14.0, 20.0,
                        20.0, 15.0, 7.0, 6.0, 15.0, 20.0,
                        20.0, 20.0, 20.0, 20.0, 20.0, 20.0]

    def start_timer(self):
        self.timer = pytimer.perf_counter()

    def clock_timer(self, show=True):
        current_time = pytimer.perf_counter() - self.timer
        if show:
            print(f"iteration time: {current_time : .2f}")
        self.start_timer()
        return current_time
