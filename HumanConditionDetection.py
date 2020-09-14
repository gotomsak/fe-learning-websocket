from imutils.video import VideoStream
import numpy as np
from imutils import face_utils
import imutils
import dlib
import cv2
from scipy.spatial import distance
import json
import pathlib
# from . import eye_open_check
import math
from keras.models import model_from_json
from sklearn import preprocessing
import io
# from PIL import Image
import base64


class HumanConditionDetection():

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            '../classification_tool/Facial-Expression-Keras-master/haarcascade_frontalface_default.xml')
        self.model = model_from_json(open(
            "../classification_tool/Facial-Expression-Keras-master/model/model.json", "r").read())
        self.model.load_weights(
            '../classification_tool/Facial-Expression-Keras-master/model/model.h5')
        self.emotions = ('Angry', 'Disgust', 'Fear', 'Happy',
                         'Neutral', 'Sad', 'Surprise')
        self.predictor = dlib.shape_predictor(
            '../classification_tool/shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
    # 人体検出

    def human_body_detection(self):
        pass

    # 顔検出
    def face_detection(self, frame):
        # face_cascade = cv2.CascadeClassifier('../classification_tool/haarcascade_frontalface_alt2.xml')
        faces = self.face_cascade.detectMultiScale(
            frame, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
        rects = self.detector(frame, 0)
        return faces, rects

    # 手検出

    def hand_detection(self):
        pass

    # 年齢推定
    def age_estimation(self):
        pass

    # 性別推定
    def sex_estimation(self):
        pass

    # 表情推定
    def expression_estimation(self, frame, face):
        distances = []
        (x, y, w, h) = face
        detected_face = frame[int(y):int(y + h), int(x):int(x + w)]
        image = imutils.resize(detected_face, width=200, height=200)
        rects = self.detector(image, 1)

        for (i, rect) in enumerate(rects):
            shape = self.predictor(image, rect)
            shape = face_utils.shape_to_np(shape)
            distances = self.euclidean_all(shape)
            # print('shape: ', shape)

        if(len(distances) != 0):
            val = distances.split(" ")[1:]
            val = np.array(val)
            val = val.astype(np.float)
            val = np.expand_dims(val, axis=1)
            minmax = preprocessing.MinMaxScaler()
            val = minmax.fit_transform(val)
            val = val.reshape(1, 4624)

            # store probabilities of 6 expressions
            predictions = self.model.predict(val)
            # find max indexed array ( 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise')
            emotion_dic = {'Angry': predictions[0][0] / 1.0 * 100, 'Disgust': predictions[0][1] / 1.0 * 100, 'Fear': predictions[0][2] / 1.0 * 100, 'Happy': predictions[0]
                           [3] / 1.0 * 100, 'Neutral': predictions[0][4] / 1.0 * 100, 'Sad': predictions[0][5] / 1.0 * 100, 'Surprise': predictions[0][6] / 1.0 * 100}
            print("Angry: %", predictions[0][0] / 1.0 * 100)
            print("Disgust: %", predictions[0][1] / 1.0 * 100)
            print("Fear: %", predictions[0][2] / 1.0 * 100)
            print("Happy: %", predictions[0][3] / 1.0 * 100)
            print("Neutral: %", predictions[0][4] / 1.0 * 100)
            print("Sad: %", predictions[0][5] / 1.0 * 100)
            print("Surprised: %", predictions[0][6] / 1.0 * 100)
            print("----------------------")
            max_index = np.argmax(predictions[0])
            emotion = self.emotions[max_index]
            print(emotion + ":" +
                  '{:2.2f}'.format(np.max(predictions[0]) / 1.0 * 100))
            return emotion_dic

    def euclidean(self, a, b):
        dist = math.sqrt(
            math.pow((b[0] - a[0]), 2) + math.pow((b[1] - a[1]), 2))
        return dist

    # calculates distances between all 68 elements
    def euclidean_all(self, a):
        distances = ""
        for i in range(0, len(a)):
            for j in range(0, len(a)):
                dist = self.euclidean(a[i], a[j])
                dist = "%.2f" % dist
                distances = distances + " " + str(dist)
        return distances

    # 視線推定
    def eyes_estimation(self):
        pass

    # 目つむり推定
    def blink_estimation(self):
        pass

    # 顔の向き推定
    def face_direction_estimation(self):
        pass

    # 顔認証
    def face_auth_estimation(self):
        pass

    def landmark_view(self):
        face_direction_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
                                          tuple(shape[48]), tuple(shape[54])])
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

    def video_read(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces, rects = self.face_detection(frame)
            # print('rects: ', rects)
            for face in faces:

                # shape = self.predictor(frame, rect)
                # shape = face_utils.shape_to_np(shape)
                self.expression_estimation(frame, face)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    # def blob_read(self, blob_data):
    #     # for i in blob_data:
    #     #     print(i)
    #     img_binarystream = io.BytesIO(blob_data)
    #     # img_pil = Image.open(img_binarystream)
    #     img_numpy = np.asarray(img_pil)
    #     frame = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
    #     faces, rects = self.face_detection(frame)
    #     for face in faces:

    #         self.expression_estimation(frame, face)

    def url_read(self, url):
        print(url)

    def base64_read(self, base64_data):
        # for i in blob_data:
        #     print(i)
        base64_data = base64_data[22:]
        base64_data = base64_data + '=' * (-len(base64_data) % 4)
        b = base64.urlsafe_b64decode(base64_data)

        #img_data = base64.b64decode(b)
        # img_binarystream = io.BytesIO(b)
        # img_pil = Image.open(img_binarystream)
        # img_numpy = np.asarray(img_pil)
        img_numpy = np.fromstring(b, np.uint8)
        frame = cv2.imdecode(img_numpy, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("nyan", frame)
        faces, rects = self.face_detection(frame)
        res = {}
        for face in faces:
            res = self.expression_estimation(frame, face)

        return res


if __name__ == "__main__":
    hcd = HumanConditionDetection()
    hcd.video_read(0)
