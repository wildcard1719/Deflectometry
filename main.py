# 한줄 x 가져오기 -> X = hilbert변환 -> x/X 페이저아크탄젠트 -> 위상펼치기 -> 기준주파수 빼기

import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
import zmq

RAW_W = 640
RAW_H = 480
image_phase_1 = np.zeros((RAW_H, RAW_W, 1), np.uint8)
image_phase_2 = np.zeros((RAW_H, RAW_W, 1), np.uint8)
image_phase_3 = np.zeros((RAW_H, RAW_W, 1), np.uint8)
image_phase_4 = np.zeros((RAW_H, RAW_W, 1), np.uint8)


IP = "192.168.35.148"
context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.setsockopt(zmq.SUBSCRIBE, b"PRV")
subscriber.setsockopt(zmq.SUBSCRIBE, b"DAT")
subscriber.setsockopt(zmq.SUBSCRIBE, b"IM1")
subscriber.setsockopt(zmq.SUBSCRIBE, b"IM2")
subscriber.setsockopt(zmq.SUBSCRIBE, b"IM3")
subscriber.setsockopt(zmq.SUBSCRIBE, b"IM4")
subscriber.connect("tcp://"+IP+":8080")
commander = context.socket(zmq.PUSH)
commander.connect("tcp://"+IP+":8081")

fig, ax = plt.subplots()
plt.ion()

freq_sample = 200
freq_cutoff = 8
coef_filter = 3
lpf = scipy.signal.firwin(coef_filter, freq_cutoff, fs=freq_sample, pass_zero=True)

VERTICAL_WAVE = False


flag_flow = "PRV"


def Preview():
    try:
        message = subscriber.recv_multipart(flags=zmq.NOBLOCK)
        header = message[0].decode()
        data = message[1]
        if header == "PRV":
            image_prv = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            cv2.imshow("Deflectometry", image_prv)
            k = cv2.waitKey(1)
            if k == ord(' '):
                flag_flow = "RCV"
                commander.send_multipart([b"CMD", "cap".encode()])
            if k == 27:
                flag_flow = "ABT"
        if header == "DAT":
            plot_data = list(data)
            ax.cla()
            #plot_data = scipy.signal.lfilter(lpf, 1.0, roi[int(D_ROI / 2), :])
            ax.plot(plot_data, 'b-')
            plt.pause(0.0000001)
            plt.show(block=False)

    except zmq.Again:  # 데이터가 없으면 계속 루프 진행
        pass


def Recv():
    global image_phase_1, image_phase_2, image_phase_3, image_phase_4
    message = subscriber.recv_multipart(flags=zmq.NOBLOCK)
    header = message[0].decode()
    data = message[1]
    if header == "IM1":
        image_phase_1 = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if header == "IM2":
        image_phase_2 = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if header == "IM3":
        image_phase_3 = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if header == "IM4":
        image_phase_4 = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        flag_flow = "PRC"


def Process():
    image = np.atan2((image_phase_4 - image_phase_2), (image_phase_1 - image_phase_3))
    cv2.imshow("Deflectometry", image)
    k = cv2.waitKey(1)
    if k == 27:
        flag_flow = "ABT"


while True:
    if flag_flow == "PRV":
        Preview()
    if flag_flow == "RCV":
        Recv()
    if flag_flow == "PRC":
        Process()
    if flag_flow == "ABT":
        break

cv2.destroyAllWindows()