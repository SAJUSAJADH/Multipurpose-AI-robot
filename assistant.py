import cv2
import numpy as np
import os
import random
import math
import sys
from collections import deque
import face_recognition
from ultralytics import YOLO
from deepface import deepface




def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2))
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2))


class AIAssistant:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.last_matches = deque(maxlen=100)

        self.class_list = self.load_class_list()
        self.detection_colors = self.generate_detection_colors()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.encode_faces()
        self.yolo_model = YOLO("weights/yolov8n.pt", "v8")

    def load_class_list(self):
        with open("utils/coco.txt", "r") as my_file:
            data = my_file.read()
        return data.split("\n")

    def generate_detection_colors(self):
        detection_colors = []
        for _ in range(len(self.class_list)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            detection_colors.append((b, g, r))
        return detection_colors

    def encode_faces(self):
        for image in os.listdir('features/faces'):
            face_image = face_recognition.load_image_file(f'features/faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

    def face_recognition_process(self, frame):
        if self.process_current_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = 'Unknown'
                confidence = '0'

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])
                    if float(confidence) < 90:
                        name = 'Unknown.0000000000.jpg'
                        confidence = 'Unknown'
                else:
                    name = 'Unknown.0000000000.jpg'
                    confidence = 'Unknown'

                name_part = name.split(".")[0]
                self.face_names.append(f'{name_part} ({confidence}) %')

        self.process_current_frame = not self.process_current_frame

        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    def detect_emotions(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    def detect_objects(self, frame):
        detect_params = self.yolo_model.predict(source=[frame], conf=0.45, save=False)
        DP = detect_params[0].numpy()
        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    self.detection_colors[int(clsID)],
                    3,
                )

                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    self.class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("Cannot open camera")

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            self.face_recognition_process(frame)
            self.detect_emotions(frame)
            self.detect_objects(frame)

            cv2.imshow("AI Assistant", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()