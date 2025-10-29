import cv2

import conf
import numpy as np
from models import ray


X_unit = np.array([1, 0, 0])
Y_unit = np.array([0, 1, 0])
Z_unit = np.array([0, 0, 1])


class Camera:
    def __init__(self, parent):
        self.location_origin = conf.CAMERA_ORIGIN
        self.location = conf.CAMERA_ORIGIN
        self.direction_origin = conf.CAMERA_DIRECTION
        self.direction = conf.CAMERA_DIRECTION
        self.resolution = conf.CAMERA_RESOLUTION
        self.aspect_ratio = self.resolution[1] / self.resolution[0]
        self.fov = np.radians(conf.CAMERA_FOV)
        self.F = conf.CAMERA_F
        self.aperture = conf.CAMERA_APERTURE

    def set_direction(self, target):
        v = target.get_location() - self.location
        self.direction = self.direction = v / np.linalg.norm(v)

    def get_location_origin(self):
        return self.location_origin

    def get_location(self):
        return self.location

    def get_width(self):
        return self.resolution[1]

    def get_height(self):
        return self.resolution[0]

    def vector_pan_tilt(self, vector, theta, phi):
        p = vector / np.linalg.norm(vector)
        u_ = np.cross(p, Z_unit)
        u = u_ / np.linalg.norm(u_)
        v_ = np.cross(u, p)
        v = v_ / np.linalg.norm(v_)

        p_rot_by_v = p * np.cos(theta) + np.cross(v, p) * np.sin(theta) + v * np.dot(v, p) * (1 - np.cos(theta))
        u_rot_by_v = u * np.cos(theta) + np.cross(v, u) * np.sin(theta) + v * np.dot(v, u) * (1 - np.cos(theta))
        p_rot_by_v_norm = p_rot_by_v / np.linalg.norm(p_rot_by_v)
        u_rot_by_v_norm = u_rot_by_v / np.linalg.norm(u_rot_by_v)
        p_rot_by_vu = p_rot_by_v_norm * np.cos(phi) + np.cross(u_rot_by_v_norm, p_rot_by_v_norm) * np.sin(phi) + u_rot_by_v_norm * np.dot(u_rot_by_v_norm, p_rot_by_v_norm) * (1 - np.cos(phi))
        return p_rot_by_vu

    def gen_ray(self, x, y):
        fov_vertical = self.fov / self.aspect_ratio
        theta = self.fov / self.resolution[1] * (x - self.resolution[1] / 2)
        phi = fov_vertical / self.resolution[0] * (self.resolution[0] / 2 - y)
        return ray.Ray(self.location, self.vector_pan_tilt(self.direction, theta, phi), 0)

