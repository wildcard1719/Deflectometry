import cv2
import zmq
import numpy as np




while True:
    try:
        buffer = socket.recv(flags=zmq.NOBLOCK)
        image_raw = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        image = cv2.resize(image_raw, (640, 480))
        cv2.imshow("stream test", image)

        if cv2.waitKey(1) == 27:
            break

    except zmq.Again:  # 데이터가 없으면 계속 루프 진행
        continue


cv2.destroyAllWindows()