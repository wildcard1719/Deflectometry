import numpy
import cv2
from sim import *

sim = Sim()

mirror = Mirror(sim)
lcd = Lcd(sim)
cam = Camera(sim)

lcd.set_direction(mirror)
cam.set_direction(mirror)

sim.render(cam)
image = sim.get_image_raw()
depth = sim.get_image_depth_raw()

cv2.imwrite('image_raw.jpg', image)
cv2.imwrite('image_depth.jpg', depth)