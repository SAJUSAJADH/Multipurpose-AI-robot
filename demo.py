import random

import cv2
import numpy as np
from ultralytics import YOLO


def object_detection():

    my_file = open("utils/coco.txt", "r")
 
    data = my_file.read()
 
    class_list = data.split("\n")
    my_file.close()

    detection_colors = []
    for i in range(len(class_list)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        detection_colors.append((b, g, r))


    model = YOLO("weights/yolov8n.pt", "v8")


    frame_wid = 640
    frame_hyt = 480

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)


        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break


        
        detect_params = model.predict(source=[frame], conf=0.45, save=False)

        DP = detect_params[0].numpy()
        print(DP)

        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                print(i)

                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

     
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

    
        cv2.imshow("ObjectDetection", frame)


        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

object_detection()