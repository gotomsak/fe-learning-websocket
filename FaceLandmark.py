import HumanConditionDetection
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


class FaceLandmark(HumanConditionDetection.HumanConditionDetection):
    def __init__(self, right_t_provisional, left_t_provisional):
        super().__init__()
        # 顔のパーツの位置をxyz軸で固定, 回転行列へ
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # 30
            (-30.0, -125.0, -30.0),  # 21
            (30.0, -125.0, -30.0),  # 22
            (-60.0, -70.0, -60.0),  # 39
            (60.0, -70.0, -60.0),  # 42
            (-40.0, 40.0, -50.0),  # 31
            (40.0, 40.0, -50.0),  # 35
            (-70.0, 130.0, -100.0),  # 48
            (70.0, 130.0, -100.0),  # 54
            (0.0, 158.0, -10.0),  # 57
            (0.0, 250.0, -50.0)  # 8
        ])

        self.angle_threshold_pitch = 12.5
        self.angle_threshold_yaw = 20
        self.angle_threshold_roll = 15
        self.max_x_diff = 0
        self.max_y_diff = 0
        self.COUNTER = 0
        self.blink_total = 0
        self.face_move_total = 0
        # すべてのフレームのカウント
        self.all_frame_cnt = 0

        # 1フレーム目の回転角
        self.fast_yaw = 0
        self.fast_pitch = 0
        self.fast_roll = 0

        # 1フレーム前の顔の位置のポイント
        self.old_points = None

        self.EYE_AR_CONSEC_FRAMES = 1

        self.right_t_provisional = right_t_provisional
        self.left_t_provisional = left_t_provisional

    # ロドリゲスポイント抽出
    def rodrigues_point(self, frame, shape):

        image_points = np.array([tuple(shape[30]),
                                 tuple(shape[21]),
                                 tuple(shape[22]),
                                 tuple(shape[39]),
                                 tuple(shape[42]),
                                 tuple(shape[31]),
                                 tuple(shape[35]),
                                 tuple(shape[48]),
                                 tuple(shape[54]),
                                 tuple(shape[57]),
                                 tuple(shape[8])], dtype='double')
        return image_points

    def get_concentration(self, move_num, max_frequency, min_frequency):
        return round((move_num - max_frequency) /
                     (min_frequency - max_frequency), 2)

    def get_weight(self, yaw, pitch, roll):
        return 1 - (((yaw / self.angle_threshold_yaw) + (pitch / self.angle_threshold_pitch) + (roll / self.angle_threshold_roll)) / 3)

    def get_rodrigues_angle(self, frame, image_points):
        size = frame.shape

        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2)

        # カメラの焦点位置, レンズの歪みを
        # 焦点距離 focal_length, 画像の中心center
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))

        # 物体の姿勢を求める 成功, 回転ベクトル, 並進ベクトル（平行移動行列？）を返す　ロドリゲスの公式
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        # 回転行列と回転ベクトルを相互に変換
        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
        mat = np.hstack((rotation_matrix, translation_vector))

        # homogeneous transformation matrix (projection matrix)　射影行列を，回転行列とカメラ行列に分解します．
        (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(
            mat)  # 回転を表す3つのオイラー角．
        # 顔の横の向き17で判定
        yaw = float(eulerAngles[1])

        # 顔の上下の向き　156以下170以上で判定
        pitch = float(eulerAngles[0])

        # 顔の回転 10で判定
        roll = float(eulerAngles[2])

        return yaw, pitch, roll

    def get_move_diff(self, image_points):
        # 1フレーム前と今のlistの差分をframe_change_listに入れた
        x_diff = 0
        y_diff = 0
        for p in range(len(image_points)):
            if type(self.old_points) == type(image_points):
                # print(int(image_points[p][0]) - int(old_points[p][0]))
                x_diff = int(image_points[p][0]) - \
                    int(self.old_points[p][0])
                y_diff = int(image_points[p][1]) - \
                    int(self.old_points[p][1])

            # cv2.circle(frame, (int(image_points[p][0]), int(
            #     image_points[p][1])), 3, (0, 0, 255), -1)

        self.old_points = image_points
        return x_diff, y_diff

    def calc_ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        eye_ear = (A + B) / (2.0 * C)
        return round(eye_ear, 3)

    def blink_count(self, left_eye_ear, right_eye_ear):
        right_t = self.right_t_provisional - 0.05
        left_t = self.left_t_provisional - 0.05
        print("right_t", right_t)
        print("right_eye_ear", right_eye_ear)
        if right_eye_ear < right_t and left_eye_ear < left_t:
            # 瞬き閾値より現在のearが下回った場合(目を閉じた時)
            self.COUNTER += 1
        # 瞬き閾値より現在のearが上回った場合(目を開けた時)
        else:
            # 　目を開けた時、カウンターが一定値以上だったら
            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.blink_total += 1

            self.COUNTER = 0

    def get_face_move(self, frame, face):
        (x, y, w, h) = face
        detected_face = frame[int(y):int(y + h), int(x):int(x + w)]
        x_diff = 0
        y_diff = 0
        pitch_diff = 0
        yaw_diff = 0
        roll_diff = 0
        # 輪郭の抽出？
        rects = self.detector(detected_face, 1)
        rect_frag = False
        for rect in rects:
            rect_frag = True
            shape = self.predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            # ロドリゲスのポイント抽出
            image_points = self.rodrigues_point(frame, shape)

            yaw, pitch, roll = self.get_rodrigues_angle(
                frame, image_points)
            print("yaw_:", yaw)
            print("pitch_:", pitch)
            print("roll_:", roll)

            if(abs(yaw) < self.angle_threshold_yaw and abs(pitch) < self.angle_threshold_pitch and abs(roll) < self.angle_threshold_roll):
                print("yaw_if: ", yaw)
                print("pitch_if: ", pitch)
                print("roll_if: ", roll)
                yaw_diff = abs(yaw)
                pitch_diff = abs(pitch)
                roll_diff = abs(roll)

            else:
                print("else: angle")

                yaw_diff = self.angle_threshold_yaw
                pitch_diff = self.angle_threshold_pitch
                roll_diff = self.angle_threshold_roll

            x_diff, y_diff = self.get_move_diff(image_points)

            left_eye = shape[42:48]

            left_eye_ear = self.calc_ear(left_eye)

            right_eye = shape[36:42]

            right_eye_ear = self.calc_ear(right_eye)

            self.blink_count(left_eye_ear, right_eye_ear)

        if rect_frag == False:
            yaw_diff = self.angle_threshold_yaw
            pitch_diff = self.angle_threshold_pitch
            roll_diff = self.angle_threshold_roll
            x_diff = self.max_x_diff
            y_diff = self.max_y_diff

        print(self.all_frame_cnt)

        if self.max_x_diff < x_diff:
            self.max_x_diff = x_diff

        if self.max_y_diff < y_diff:
            self.max_y_diff = y_diff

        return x_diff, y_diff, yaw_diff, pitch_diff, roll_diff

    def main_face_landmark(self, base64_data):
        self.all_frame_cnt += 1

        base64_data = base64_data[22:]
        base64_data = base64_data + '=' * (-len(base64_data) % 4)
        b = base64.urlsafe_b64decode(base64_data)
        img_numpy = np.fromstring(b, np.uint8)
        frame = cv2.imdecode(img_numpy, cv2.COLOR_BGR2GRAY)

        faces, rects = self.face_detection(frame)
        res_x = 0
        res_y = 0
        pitch_diff = 0
        yaw_diff = 0
        roll_diff = 0
        # res = {}
        c1 = 0
        c2 = 0
        c3 = 0

        w = 1
        for face in faces:
            res_x, res_y, yaw_diff, pitch_diff, roll_diff = self.get_face_move(
                frame, face)
            # c1 = self.get_concentration(pitch_diff+yaw_diff+roll_diff,self.t)
        if (len(faces) != 0):
            c2 = self.get_concentration(res_x + res_y, 100, 2)
            self.face_move_total += res_x + res_y
            w = self.get_weight(yaw_diff, pitch_diff, roll_diff)
        res = {
            "blink": self.blink_total,
            "face_move": self.face_move_total,
            "w": w,
            "c1": c1,
            "c2": c2,
            "c3": c3,
        }
        print("c2: ", c2)
        print("w: ", w)
        print("blink_count", self.blink_total)

        return res
