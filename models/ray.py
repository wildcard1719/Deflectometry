import cv2

import conf
import numpy as np


X_unit = np.array([1, 0, 0])
Y_unit = np.array([0, 1, 0])
Z_unit = np.array([0, 0, 1])


class Ray:
    def __init__(self, origin, direction, depth):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)
        self.depth = depth
        self.hit = 0
        self.reflected = 0
        self.intensity = 0

    def get_origin(self):
        return self.origin

    def get_direction(self):
        return self.direction

    def depth_add(self, d):
        self.depth += d

    def get_depth(self):
        return self.depth

    def set_hit(self):
        self.hit = 1

    def set_reflected(self):
        self.reflected = 1

    def reset_hit(self):
        self.hit = 0

    def get_hit(self):
        return self.hit

    def get_reflected(self):
        return self.reflected

    def shade(self, intensity):
        self.intensity = intensity

    def get_shade(self):
        return self.intensity
