# 한줄 x 가져오기 -> X = hilbert변환 -> x/X 페이저아크탄젠트 -> 위상펼치기 -> 기준주파수 빼기

import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
import zmq

RAW_W = 1920
RAW_H = 1080
IMAGE_W = 1080

image_phase_1 = np.zeros((RAW_H, RAW_W, 1), np.float16)
image_phase_2 = np.zeros((RAW_H, RAW_W, 1), np.float16)
image_phase_3 = np.zeros((RAW_H, RAW_W, 1), np.float16)
image_phase_4 = np.zeros((RAW_H, RAW_W, 1), np.float16)

image_angle = np.zeros((RAW_H, RAW_W, 1), np.float16)
image_unwrap = np.zeros((RAW_H, RAW_W, 1), np.float16)
image_ref = np.zeros((RAW_H, RAW_W, 1), np.float16)
image_delta = np.zeros((RAW_H, RAW_W, 1), np.float16)
image_height = np.zeros((RAW_H, RAW_W, 1), np.float16)


IP = "192.168.0.6"
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

def norm(image, a, b, tp:type):
    image_min = image.min()
    image_max = image.max()
    return ( a + (((image - image_min)*(b-a)) / (image_max - image_min)).astype(tp) )

def square(image):
    h = image.shape(0)
    w = image.shape(1)
    return image[h//2 - IMAGE_W//2 : h//2 + IMAGE_W//2, w//2 - IMAGE_W//2 : w//2 + IMAGE_W//2]


def Preview():
    global flag_flow
    try:
        message = subscriber.recv_multipart(flags=zmq.NOBLOCK)
        header = message[0].decode()
        data = message[1]
        if header == "PRV":
            image_prv = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            cv2.imshow("Deflectometry", image_prv)
            k = cv2.waitKey(1) & 0xFF
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
    global flag_flow, image_phase_1, image_phase_2, image_phase_3, image_phase_4
    message = subscriber.recv_multipart()
    header = message[0].decode()
    data = message[1]
    if header == "IM1":
        print("[+] Recv Start")
        image_phase_1 = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if header == "IM2":
        image_phase_2 = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if header == "IM3":
        image_phase_3 = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if header == "IM4":
        print("[+] Recv Done")
        image_phase_4 = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        flag_flow = "PRC"


def Process():
    global flag_flow, image_angle, image_unwrap, image_ref, image_delta
    image_1 = cv2.normalize(image_phase_1, None, 0, 255, cv2.NORM_MINMAX)
    image_2 = cv2.normalize(image_phase_2, None, 0, 255, cv2.NORM_MINMAX)
    image_3 = cv2.normalize(image_phase_3, None, 0, 255, cv2.NORM_MINMAX)
    image_4 = cv2.normalize(image_phase_4, None, 0, 255, cv2.NORM_MINMAX)

    image_angle = (np.angle(((image_1-128.0) - (image_3-128.0))/2 + 1j * ((image_2-128.0) - (image_4-128.0))/2) + np.pi).astype(np.float32)
    #image_angle = (np.angle((image_1-128.0) + 1j * (image_2-128.0)) + np.pi).astype(np.float32)
    image_angle_norm = norm(image_angle, 0, 255, np.uint8)

    for i in range(image_angle.shape[1]):
        image_ref[:, i] = 255.0 - ((float(i)/float(image_angle.shape[1]))* 255.0)

    for i in range(image_angle.shape[0]):
        line = image_angle[i, :]
        line_unwrap = np.unwrap(line, period=np.pi*2)
        line_norm = norm(line_unwrap, 0, 255, np.float16)
        image_unwrap[i, :] = line_norm.reshape(-1, 1)

    image_delta = norm(((image_unwrap - image_ref) +128.0), 0, 255, np.float16)

    image_height = np.cumsum((image_delta-128.0)*0.002, axis=1)+128

    plt.clf()
    plt.imshow(image_height, cmap='jet')
    plt.colorbar()
    plt.pause(0.1)
    plt.show(block=False)

    cv2.imshow("Deflectometry", cv2.resize(image_delta.astype(np.uint8), (1920//2, 1080//2)))
    cv2.imwrite("pics/raw.png", image_1.astype(np.uint8))
    cv2.imwrite("pics/phase.png", image_angle_norm.astype(np.uint8))
    cv2.imwrite("pics/unwrap.png", image_unwrap.astype(np.uint8))
    cv2.imwrite("pics/delta.png", image_delta.astype(np.uint8))
    cv2.imwrite("pics/height.png", image_height.astype(np.uint8))
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