import cv2

import conf
import numpy as np
from scipy.optimize import root_scalar


X_unit = np.array([1, 0, 0])
Y_unit = np.array([0, 1, 0])
Z_unit = np.array([0, 0, 1])


class Sim:
    def __init__(self):
        self.objects = []

    def deflectometry(self, cam, lcd):
        vertical = 1
        lcd.set_vertical(vertical)

        image_green, mask = self.get_mask(cam, lcd)

        image_angle_1 = self.get_phase_image(cam, lcd)
        image_angle_1_masked = np.ma.array(image_angle_1, mask=mask)
        lcd.move()
        image_angle_2 = self.get_phase_image(cam, lcd)
        image_angle_2_masked = np.ma.array(image_angle_2, mask=mask)

        phase_delta = abs(self.phase_distance(image_angle_1_masked, image_angle_2_masked))
        phase_delta_norm = self.norm(phase_delta, 0, 255, np.float16)

        cv2.imwrite("./sim_pics/phase_delta.jpg", phase_delta_norm.astype(np.uint8))


    def get_mask(self, cam, lcd):
        lcd.green = 1
        image_green, _ = self.render(cam)
        lcd.green = 0
        mask = np.zeros_like(image_green)
        mask[image_green == 0] = True
        return image_green, mask


    def get_phase_image(self, cam, lcd):
        lcd.set_phase(0)
        image_phase_1, _ = self.render(cam)
        cv2.imwrite("./sim_pics/raw.jpg", image_phase_1)
        lcd.set_phase(np.pi / 2)
        image_phase_2, _ = self.render(cam)
        lcd.set_phase(np.pi)
        image_phase_3, _ = self.render(cam)
        lcd.set_phase(np.pi / 2 * 3)
        image_phase_4, _ = self.render(cam)

        image_angle = (np.angle(((image_phase_1 - 128.0) - (image_phase_3 - 128.0)) / 2 + 1j * (
                    (image_phase_2 - 128.0) - (image_phase_4 - 128.0)) / 2)).astype(np.float16)
        image_angle_norm = self.norm(image_angle, 0, 255, np.float16)
        cv2.imwrite("./sim_pics/phase.jpg", image_angle_norm)
        return image_angle



    def phase_distance(self, image_1, image_2):
        delta = image_2 - image_1
        return (delta + np.pi) % (2 * np.pi) - np.pi


    def gen_ref_phase_image(self, image, vertical):
        ys, xs, _ = np.where(image == 255)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        image_ref = np.zeros((h, w, 1), np.float16)
        image_ref_reverse = np.zeros((h, w, 1), np.float16)
        if vertical == 0:
            for i in range(w):
                image_ref[:, i] = ((float(i) / float(w)) * 255.0)
                image_ref_reverse[:, i] = 255.0 - ((float(i) / float(w)) * 255.0)
        else:
            for i in range(h):
                image_ref[i, :] = ((float(i) / float(h)) * 255.0)
                image_ref_reverse[i, :] = 255.0 - ((float(i) / float(h)) * 255.0)
        return image_ref, image_ref_reverse, (x_min, x_max, y_min, y_max)


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

        return image_raw.astype(np.uint8), image_depth_raw


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
        cv2.imwrite("./sim_pics/"+name+".jpg", image.astype(np.uint8))


    def norm(self, image, a, b, tp: type):
        image_min = image.min()
        image_max = image.max()
        return (a + (((image - image_min) * (b - a)) / (image_max - image_min)).astype(tp))


    def get_image_raw(self):
        return self.image_raw


    def get_image_depth_raw(self):
        return self.image_depth_raw







