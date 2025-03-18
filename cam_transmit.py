import numpy as np
import cv2
import zmq
import time
import sinegen

RES = (1280, 720)
WAVE_NUM = 50
DELAY = 0.1

flag_flow = "RDY"

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:8080")

collector = context.socket(zmq.PULL)
collector.bind("tcp://*:8081")


cv2.namedWindow("Deflectometry", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Deflectometry", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)

def WaitRequest():
    global flag_flow
    try:
        message = collector.recv_multipart(flags=zmq.NOBLOCK)
        header = message[0].decode()
        data = message[1]
        if header == "CMD":
            if data =="cap":
                flag_flow = "CAP"

    except zmq.Again:
        cv2.imshow("Deflectometry", sinegen.SinGen(0, RES, WAVE_NUM, False))
        image_prv = np.zeros((100, 100, 1), np.uint8)  # change for small cam image
        data_plot = image_prv[image_prv.shape[0]//2, :].reshape(image_prv.shape[1]).tolist()
        publisher.send_multipart([b"PRV", image_prv.tobytes()])
        publisher.send_multipart([b"DAT", bytes(data_plot)])


def Capture():
    global flag_flow

    cv2.imshow("Deflectometry", sinegen.SinGen(0, RES, WAVE_NUM, False))
    time.sleep(DELAY)
    _, image_phase_1 = cap.read()
    time.sleep(DELAY)

    cv2.imshow("Deflectometry", sinegen.SinGen(np.pi/2, RES, WAVE_NUM, False))
    time.sleep(DELAY)
    _, image_phase_2 = cap.read()
    time.sleep(DELAY)

    cv2.imshow("Deflectometry", sinegen.SinGen(np.pi, RES, WAVE_NUM, False))
    time.sleep(DELAY)
    _, image_phase_3 = cap.read()
    time.sleep(DELAY)

    cv2.imshow("Deflectometry", sinegen.SinGen(np.pi*3/2, RES, WAVE_NUM, False))
    time.sleep(DELAY)
    _, image_phase_4 = cap.read()
    time.sleep(DELAY)

    publisher.send_multipart([b"IM1", image_phase_1.tobytes()])
    publisher.send_multipart([b"IM2", image_phase_2.tobytes()])
    publisher.send_multipart([b"IM3", image_phase_3.tobytes()])
    publisher.send_multipart([b"IM4", image_phase_4.tobytes()])

    flag_flow = "RDY"


while True:
    if flag_flow == "RDY":
        WaitRequest()

    if flag_flow == "CAP":
        Capture()
