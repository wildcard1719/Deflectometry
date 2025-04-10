import numpy as np

yes = 1
no = 0

### CAMERA CONFIG ###
CAMERA_ORIGIN = np.array([100, 0, 100])  # [mm]
CAMERA_DIRECTION = 'params'  # [unit vector] ( 'mirror' or 'lcd' or np.array([x, y, z]) )
CAMERA_RESOLUTION = np.array([200, 200])  # [h, w]
CAMERA_FOV = 30  # [deg]
CAMERA_F = 17.53  # [mm]
CAMERA_APERTURE = 1/2  # [F/x]


### MIRROR CONFIG ###
MIRROR_ORIGIN = np.array([0, 0, 0])  # [mm]
MIRROR_DIRECTION = np.array([0, 0, 1])  # [unit vector]
MIRROR_HEIGHT = 15  # [mm]
MIRROR_RADIUS = 25  # [mm]
MIRROR_HOLE_RADIUS = 10  # [mm]
MIRROR_CURVE_C = 100  # [mm] ( convex->negative / concave->positive )
MIRROR_CURVE_K = -1


### LCD CONFIG ###
LCD_ORIGIN = np.array([-100, 0, 100])  # [mm]
LCD_DIRECTION = 'params'  # [unit vector]] ( 'mirror' or 'lcd' or np.array([x, y, z]) )
LCD_WIDTH = 200  # [mm]
LCD_HEIGHT = 120  # [mm]
STRIPE_WAVE_LENGTH = 10  # [mm]


### SIMULATION SETTING ###
MIRROR_HEIGHT_ERROR_ENABLE = no
MIRROR_HEIGHT_ERROR = 1  # [mm]
MIRROR_HEIGHT_ERROR_DIV = 100

MIRROR_EULER_ERROR_ENABLE = no
MIRROR_EULER_ERROR = np.array([0.1, 0.1, 0])  # [rad]
MIRROR_EULER_ERROR_DIV = np.array([100, 100, 1])  # iteration

LCD_EULER_ERROR_ENABLE = no
LCD_EULER_ERROR = np.array([0.1, 0.1, 0.1])  # [rad]
LCD_EULER_ERROR_DIV = np.array([100, 100, 100])  # iteration