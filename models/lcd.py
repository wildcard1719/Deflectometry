import cv2

import conf
import numpy as np


X_unit = np.array([1, 0, 0])
Y_unit = np.array([0, 1, 0])
Z_unit = np.array([0, 0, 1])


class Lcd:
    def __init__(self, parent):
        self.object_info = 'lcd'
        self.location_origin = conf.LCD_ORIGIN
        self.location = conf.LCD_ORIGIN
        self.move_delta = conf.LCD_MOVE_VECTOR
        self.direction_origin = conf.LCD_DIRECTION
        self.direction = conf.LCD_DIRECTION
        self.width = conf.LCD_WIDTH
        self.height = conf.LCD_HEIGHT
        self.stripe_wave_length = conf.STRIPE_WAVE_LENGTH
        self.stripe_vertical = 1
        self.stripe_phase = 0
        self.green = 0
        self.parent = parent
        self.parent.objects.append(self)

    def get_info(self):
        return self.object_info

    def get_location(self):
        return self.location

    def move(self):
        self.location += self.move_delta

    def set_direction(self, target):
        v = target.get_location_origin() - self.location
        self.direction = v / np.linalg.norm(v)

    def set_phase(self, phase):
        self.stripe_phase = phase

    def set_vertical(self, vertical):
        self.stripe_vertical = vertical

    def intersect(self, ray):
        q = self.location
        N = self.direction
        plane_u = np.cross(Z_unit, N)
        plane_u_unit = plane_u / np.linalg.norm(plane_u)
        plane_v = np.cross(N, plane_u_unit)
        plane_v_unit = plane_v / np.linalg.norm(plane_v)
        v = ray.get_direction()
        p_0 = ray.get_origin()
        if np.dot(N, v) >= 0:
            return -1
        t = np.dot(N, (q - p_0)) / np.dot(N, v)
        p = p_0 + v * t
        d = p - q
        a = np.dot(d, plane_u_unit)
        b = np.dot(d, plane_v_unit)
        if abs(a) > self.width // 2:
            return -1
        if abs(b) > self.height // 2:
            return -1

        if self.stripe_vertical == 0:
            a_edge = a + self.width // 2
            i = (np.sin(self.stripe_phase + np.pi * 2 * ((a_edge % self.stripe_wave_length) / self.stripe_wave_length)) + 1) * 127.5
        if self.stripe_vertical == 1:
            b_edge = b + self.height // 2
            i = (np.sin(self.stripe_phase + np.pi * 2 * ((b_edge % self.stripe_wave_length) / self.stripe_wave_length)) + 1) * 127.5

        if self.green == 1:
            i = 255

        ray.depth_add(np.linalg.norm(v*t))
        ray.shade(i)
        ray.set_hit()

        return 1

