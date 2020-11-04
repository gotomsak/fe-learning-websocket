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
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
import base64
import FaceLandmark


class GetFrequency(FaceLandmark.FaceLandmark):

    def __init__(self, right_t_provisional, left_t_provisional):
        super().__init__(right_t_provisional, left_t_provisional)

    def main_get_frequency(self, base64_data):
        base64_data = base64_data[22:]
        base64_data = base64_data + '=' * (-len(base64_data) % 4)
        b = base64.urlsafe_b64decode(base64_data)
        img_numpy = np.fromstring(b, np.uint8)
        frame = cv2.imdecode(img_numpy, cv2.COLOR_BGR2GRAY)

        # ランドマーク抽出
        rects = self.detector(frame, 0)
        blink_bool = False
        face_move = 0
        for rect in rects:
            shape = self.predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            # 顔の動きの座標とangleの差分を返す
            res_x, res_y, yaw_diff, pitch_diff, roll_diff = self.get_face_move(
                frame, rect, shape)

            left_eye = shape[42:48]

            left_eye_ear = self.calc_ear(left_eye)

            right_eye = shape[36:42]

            right_eye_ear = self.calc_ear(right_eye)

            blink_bool = self.blink_count(left_eye_ear, right_eye_ear)

            face_move = res_x + res_y

        return {
            "blink": blink_bool,
            "face_move": face_move
        }
