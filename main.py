import numpy as np
import cv2
from models import sim
from models import camera
from models import mirror
from models import lcd

sim = sim.Sim()

mirror = mirror.Mirror(sim)
lcd = lcd.Lcd(sim)
cam = camera.Camera(sim)

lcd.set_direction(mirror)
cam.set_direction(mirror)

image = sim.deflectometry(cam, lcd)
#image, _ = sim.render(cam)
cv2.imwrite('./sim_pics/image.jpg', image)