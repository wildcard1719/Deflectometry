import numpy as np
import cv2


def SinGen(phase_offset, res:tuple, num_of_wave, vertical=False):
    image = np.zeros((res[1], res[0], 1), np.uint8)
    if vertical == False:
        n = res[1]
        m = res[0]
    else:
        n = res[0]
        m = res[1]
    wave_len = n / num_of_wave

    for i in range(m):
        amplitude = ((np.sin(phase_offset + np.pi*2 * ((i % wave_len) / wave_len))) * 127) + 128
        if vertical == False:
            image[:, i] = amplitude
        else:
            image[i, :] = amplitude
    return image

# cv2.imwrite("sin.jpg", SinGen(-2*np.pi, (2000, 2000),  2, 1))


print("[+] Generate done")
