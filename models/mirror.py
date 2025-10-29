import cv2

import conf
import numpy as np
from models import ray
from scipy.optimize import root_scalar


X_unit = np.array([1, 0, 0])
Y_unit = np.array([0, 1, 0])
Z_unit = np.array([0, 0, 1])


class Mirror:
    def __init__(self, parent):
        self.object_info = 'mirror'
        self.location_origin = conf.MIRROR_ORIGIN
        self.location = conf.MIRROR_ORIGIN
        self.direction_origin = conf.MIRROR_DIRECTION
        self.direction = conf.MIRROR_DIRECTION
        self.height = conf.MIRROR_HEIGHT
        self.radius = conf.MIRROR_RADIUS
        self.hole_radius = conf.MIRROR_HOLE_RADIUS
        self.curve_c = conf.MIRROR_CURVE_C
        self.curve_k = conf.MIRROR_CURVE_K
        self.parent = parent
        self.parent.objects.append(self)

    def get_info(self):
        return self.object_info

    def get_location_origin(self):
        return self.location_origin

    def get_location(self):
        return self.location


    def f_asphere(self, x, y, R, k):
        r2 = x ** 2 + y ** 2
        sqrt_term = np.sqrt(1 - (1 + k) * (r2 / R ** 2))
        z_conic = r2 / (R * (1 + sqrt_term))
        return z_conic

    def F(self, t):
        x = self.ray.get_origin()[0] - self.location[0] + t * self.ray.get_direction()[0]
        y = self.ray.get_origin()[1] - self.location[1] + t * self.ray.get_direction()[1]
        z = self.ray.get_origin()[2] - self.location[2] + t * self.ray.get_direction()[2]
        R = self.curve_c
        k = self.curve_k
        return z - self.f_asphere(x, y, R, k)

    def grad_f_asphere(self, x, y, R, k, eps=1e-6):
        fx = (self.f_asphere(x + eps, y, R, k) - self.f_asphere(x - eps, y, R, k)) / (2 * eps)
        fy = (self.f_asphere(x, y + eps, R, k) - self.f_asphere(x, y - eps, R, k)) / (2 * eps)
        return fx, fy

    def get_normal_line(self, x, y, R, k):
        fx, fy = self.grad_f_asphere(x, y, R, k)
        n = np.array([-fx, -fy, 1.0])
        return n / np.linalg.norm(n)

    def line_2_point_distance(self, p, l0, v):
        cross = np.cross(p - l0, v)
        return np.linalg.norm(cross) / np.linalg.norm(v)

    def intersect(self, ray_):
        self.ray = ray_
        v = ray_.get_direction()
        p_0 = ray_.get_origin()

        try:
            sol = root_scalar(self.F, bracket=[10, 1000], method='brentq')
        except:
            return -1
        if sol.converged:
            t_hit = sol.root
            p = p_0 + v * t_hit
            if self.line_2_point_distance(p, self.location, self.direction) > self.radius or self.line_2_point_distance(p, self.location, self.direction) < self.hole_radius:
                return -1
            p_local = p - self.location
            nn = self.get_normal_line(p_local[0], p_local[1], self.curve_c, self.curve_k)
            v_norm = v / np.linalg.norm(v)
            v_reflect = (v_norm - 2 * np.dot(v_norm, nn) * nn)
            ray_.set_hit()
            return ray.Ray(p, v_reflect, np.linalg.norm(v * t_hit))
        else:
            return -1
