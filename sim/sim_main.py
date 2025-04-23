import numpy
import cv2
from sim import *

sim = Sim()

mirror = Mirror(sim)
lcd = Lcd(sim)
cam = Camera(sim)

#lcd.set_direction(mirror)
cam.set_direction(mirror)

image = sim.deflectometry(cam, lcd)
#image, _ = sim.render(cam)
cv2.imwrite('./sim/sim_pics/image.jpg', image)