import cv2

import conf
import numpy as np
from scipy.optimize import root_scalar
from skimage.restoration import unwrap_phase


X_unit = np.array([1, 0, 0])
Y_unit = np.array([0, 1, 0])
Z_unit = np.array([0, 0, 1])


class Sim:
    def __init__(self):
        self.objects = []

    def deflectometry(self, cam, lcd):
        vertical = 1
        lcd.set_vertical(vertical)
        lcd.green = 1
        image_green, _ = self.render(cam)
        lcd.green = 0
        mask = np.zeros_like(image_green)
        mask[image_green == 0] = True

        image_ref = np.zeros_like(image_green)
        image_ref_center, (x_min, x_max, y_min, y_max) = self.gen_ref_phase_image(image_green, vertical)
        image_ref[y_min:y_max+1, x_min:x_max+1] = image_ref_center
        cv2.imwrite("./sim/sim_pics/ref.jpg", image_ref)

        lcd.set_phase(0)
        image_phase_1, _ = self.render(cam)
        cv2.imwrite("./sim/sim_pics/raw.jpg", image_phase_1)
        lcd.set_phase(np.pi / 2)
        image_phase_2, _ = self.render(cam)
        lcd.set_phase(np.pi)
        image_phase_3, _ = self.render(cam)
        lcd.set_phase(np.pi / 2 * 3)
        image_phase_4, _ = self.render(cam)

        image_angle = (np.angle(((image_phase_1 - 128.0) - (image_phase_3 - 128.0)) / 2 + 1j * (
                    (image_phase_2 - 128.0) - (image_phase_4 - 128.0)) / 2)).astype(np.float16)

        image_angle_masked = np.ma.array(image_angle, mask=mask)

        image_unwrap = unwrap_phase(image_angle_masked)
        image_unwrap_norm = self.norm(image_unwrap, 0, 255, np.float16)
        cv2.imwrite("./sim/sim_pics/unwrap.jpg", image_unwrap_norm)

        image_delta = image_unwrap_norm - image_ref + 128
        image_delta_norm = self.norm(image_delta, 0, 255, np.float16)
        cv2.imwrite("./sim/sim_pics/delta.jpg", image_delta_norm)

        return image_delta_norm.astype(np.uint8)

    def gen_ref_phase_image(self, image, vertical):
        ys, xs, _ = np.where(image == 255)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        image_ref = np.zeros((h, w, 1), np.float16)
        if vertical == 0:
            for i in range(w):
                image_ref[:, i] = ((float(i) / float(w)) * 255.0)
        else:
            for i in range(h):
                image_ref[i, :] = ((float(i) / float(h)) * 255.0)
        return image_ref, (x_min, x_max, y_min, y_max)

    def render(self, cam):
        w = cam.get_width()
        h = cam.get_height()
        p = 0
        image_raw = np.zeros((h, w, 1), np.float16)
        image_depth_raw = np.zeros((h, w, 1), np.float16)

        for x in range(w):
            for y in range(h):
                p += 1
                ray = cam.gen_ray(x, y)
                image_raw[y, x], image_depth_raw[y, x] = self.trace(ray)
            print(str(round(p / (w * h) * 100, 1)) + "%")

        return image_raw, image_depth_raw


    def trace(self, ray):
        shade = 0
        depth = 0
        for obj in self.objects:
            ret = obj.intersect(ray)
            if ray.get_hit() == 1:
                if ret == 1:
                    shade = ray.get_shade()
                    depth = ray.get_depth()
                elif ret == -1:
                    shade = 0
                    depth = 0
                else:
                    ray.reset_hit()
                    ret.set_reflected()
                    if ray.get_reflected() == 0:
                        shade, depth = self.trace(ret)
        return shade, depth

    def save(self, image, name):
        mc = str(conf.MIRROR_CURVE_C)
        mk = str(conf.MIRROR_CURVE_K)
        co_1 = str(conf.CAMERA_ORIGIN[0])
        co_2 = str(conf.CAMERA_ORIGIN[1])
        co_3 = str(conf.CAMERA_ORIGIN[2])
        cd_1 = str(conf.CAMERA_DIRECTION[0])
        cd_2 = str(conf.CAMERA_DIRECTION[1])
        cd_3 = str(conf.CAMERA_DIRECTION[2])

        lo_1 = str(conf.LCD_ORIGIN[0])
        lo_2 = str(conf.LCD_ORIGIN[1])
        lo_3 = str(conf.LCD_ORIGIN[2])
        ld_1 = str(conf.LCD_DIRECTION[0])
        ld_2 = str(conf.LCD_DIRECTION[1])
        ld_3 = str(conf.LCD_DIRECTION[2])
        cv2.imwrite("./sim/sim_pics/"+name+".jpg", image.astype(np.uint8))


    def norm(self, image, a, b, tp: type):
        image_min = image.min()
        image_max = image.max()
        return (a + (((image - image_min) * (b - a)) / (image_max - image_min)).astype(tp))

    def get_image_raw(self):
        return self.image_raw

    def get_image_depth_raw(self):
        return self.image_depth_raw


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
        return Ray(self.location, self.vector_pan_tilt(self.direction, theta, phi), 0)


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

    def intersect(self, ray):
        self.ray = ray
        v = ray.get_direction()
        p_0 = ray.get_origin()

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
            ray.set_hit()
            return Ray(p, v_reflect, np.linalg.norm(v * t_hit))
        else:
            return -1


class Lcd:
    def __init__(self, parent):
        self.object_info = 'lcd'
        self.location_origin = conf.LCD_ORIGIN
        self.location = conf.LCD_ORIGIN
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

