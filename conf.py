import numpy as np

yes = 1
no = 0

### CAMERA CONFIG ###
CAMERA_ORIGIN = np.array([-200, 0, 200])  # [mm]
CAMERA_DIRECTION = np.array([-1, 0, 0])  # [unit vector]
CAMERA_RESOLUTION = np.array([100, 100])  # [h, w]
CAMERA_FOV = 18  # [deg]sim/
CAMERA_F = 17.53  # [mm]
CAMERA_APERTURE = 1/2  # [F/x]


### MIRROR CONFIG ###
MIRROR_ORIGIN = np.array([0, 0, 0])  # [mm]
MIRROR_DIRECTION = np.array([0, 0, 1])  # [unit vector]
MIRROR_HEIGHT = 15  # [mm]
MIRROR_RADIUS = 25  # [mm]
MIRROR_HOLE_RADIUS = 10  # [mm]
MIRROR_CURVE_C = 150  # [mm] ( convex->negative / concave->positive )
MIRROR_CURVE_K = -1.244


### LCD CONFIG ###
LCD_ORIGIN = np.array([150, 0, 150])  # [mm]
LCD_MOVE_VECTOR = np.array([3, 0, 3])  # [mm]
LCD_DIRECTION = np.array([-1, 0, -1])  # [unit vector]
LCD_WIDTH = 300  # [mm]
LCD_HEIGHT = 1000  # [mm]
STRIPE_WAVE_LENGTH = 15  # [mm]


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