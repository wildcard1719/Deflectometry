import numpy as np
import cv2
import zmq
import time
import sinegen

SENSOR_ID = '0'
W_RAW = 1920
H_RAW = 1080
W_ROI = 300
H_ROI = 300

FRAMERATE = 2
CAM_EXPOSURE = '"100000000 100000000"'
CAM_DIGITAL_GAIN = '"1 1"'
CAM_GAIN = '"1 1"'

gst_pipeline = ("nvarguscamerasrc sensor-id="+SENSOR_ID+" wbmode=0 aeantibanding=0 tnr-mode=1 ispdigitalgainrange="+CAM_DIGITAL_GAIN+" exposuretimerange="+CAM_EXPOSURE+" ! video/x-raw(memory:NVMM), width=(int)"+str(W_RAW)+", height=(int)"+str(H_RAW)+", format=(string)NV12, framerate=(fraction)"+str(FRAMERATE)+"/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)


RES = (1280, 720)
WAVE_NUM = 30
DELAY = 0.2

flag_flow = "RDY"

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:8080")

collector = context.socket(zmq.PULL)
collector.bind("tcp://*:8081")

image = np.zeros((H_RAW, W_RAW, 1), np.uint8)

cv2.namedWindow("Deflectometry", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Deflectometry", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def WaitRequest():
    global image, flag_flow
    try:
        message = collector.recv_multipart(flags=zmq.NOBLOCK)
        header = message[0].decode()
        data = message[1]
        if header == "CMD":
            if data.decode() =="cap":
                flag_flow = "CAP"

    except zmq.Again:
        cv2.imshow("Deflectometry", sinegen.SinGen(0, RES, WAVE_NUM, False))
        k = cv2.waitKey(1)
        if k == 27:
            flag_flow = "ABT"
        image_prv = image
        _, image_prv_zip = cv2.imencode('.png', cv2.resize(image_prv, (320, 180)))

        data_plot = image_prv[image_prv.shape[0]//2, :].reshape(image_prv.shape[1]).tolist()
        publisher.send_multipart([b"PRV", image_prv_zip.tobytes()])
        publisher.send_multipart([b"DAT", bytes(data_plot)])

phase_status = 1
phase_timer = 0
phase_timer_max = 10

def Capture():
    global image, flag_flow, phase_status, phase_timer
    
    if phase_status == 1:
        if phase_timer == 0:
            image_phase_1 = image
            _, image_encode_1 = cv2.imencode('.png', image_phase_1)
            publisher.send_multipart([b"IM1", image_encode_1.tobytes()])
            cv2.imshow("Deflectometry", sinegen.SinGen(np.pi/2, RES, WAVE_NUM, False))
        cv2.waitKey(1)
        phase_timer += 1
        if phase_timer >= phase_timer_max:
            phase_timer = 0
            phase_status = 2

    elif phase_status == 2:
        if phase_timer == 0:
            image_phase_2 = image
            _, image_encode_2 = cv2.imencode('.png', image_phase_2)
            publisher.send_multipart([b"IM2", image_encode_2.tobytes()])
            cv2.imshow("Deflectometry", sinegen.SinGen(np.pi, RES, WAVE_NUM, False))
        cv2.waitKey(1)
        phase_timer += 1
        if phase_timer >= phase_timer_max:
            phase_timer = 0
            phase_status = 3

    elif phase_status == 3:
        if phase_timer == 0:
            image_phase_3 = image
            _, image_encode_3 = cv2.imencode('.png', image_phase_3)
            publisher.send_multipart([b"IM3", image_encode_3.tobytes()])
            cv2.imshow("Deflectometry", sinegen.SinGen(np.pi*3/2, RES, WAVE_NUM, False))
        cv2.waitKey(1)
        phase_timer += 1
        if phase_timer >= phase_timer_max:
            phase_timer = 0
            phase_status = 4

    elif phase_status == 4:
        if phase_timer == 0:
            image_phase_4 = image
            _, image_encode_4 = cv2.imencode('.png', image_phase_4)
            publisher.send_multipart([b"IM4", image_encode_4.tobytes()])
        cv2.waitKey(1)  
        phase_timer += 1
        if phase_timer >= phase_timer_max:
            phase_timer = 0
            phase_status = 1
            flag_flow = "RDY"



while True:
    _, image_color = cap.read()
    image = cv2.flip(cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY), -1)


    if flag_flow == "RDY":
        WaitRequest()

    if flag_flow == "CAP":
        Capture()

    if flag_flow == "ABT":
        break

cv2.destroyAllWindows()
