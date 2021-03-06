
# from typing_extensions import final
from imutils.video import VideoStream
import numpy as np
from imutils import face_utils

import dlib
import cv2
from scipy.spatial import distance
import json

# from . import eye_open_check


import base64
import datetime
import os
# from HumanConditionDetection import HumanConditionDetection


class FaceLandmark:
    def __init__(self, right_t_provisional, left_t_provisional):
        # super().__init__()
        # def __init__(self):
        self.predictor = dlib.shape_predictor(
            './classification_tool/shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
        self.face_cascade = cv2.CascadeClassifier(
            './classification_tool/Facial-Expression-Keras-master/haarcascade_frontalface_default.xml')

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

        # 5秒おきの瞬きした回数の全てのリスト
        self.all_blink_list = []

        # 5秒おきのangleの動きの合計の全てのリスト
        self.all_angle_list = []
        # 5秒おきの顔の動きの合計の全てのリスト
        self.all_face_move_list = []

        # 1フレーム目の回転角
        self.fast_yaw = 0
        self.fast_pitch = 0
        self.fast_roll = 0

        # 1フレーム前の顔の位置のポイント
        self.old_points = None

        self.EYE_AR_CONSEC_FRAMES = 1
        # 目のしきい値？
        self.right_t_provisional = right_t_provisional
        self.left_t_provisional = left_t_provisional

        self.angle_threshold_up = 0
        self.angle_threshold_down = 0

        # 5秒間(0.5sオキ)の顔のアングルの数値のリスト
        self.section_5_angle_list = []

        # 5秒間(0.5sオキ)のtrue,falseのリスト
        self.section_5_blink_list = []

        # 5秒間(0.5sオキ)の顔の動きの数値のリスト
        self.section_5_face_move_list = []

        self.save_dir_path = ""
        self.full_save_file_path = ""
        self.max_blink_freq = 0
        self.min_blink_freq = 0
        self.max_face_move_freq = 0
        self.min_face_move_freq = 0
        self.final_list = []
        self.lastEmotion = []
        self.angry_list = []
        self.disgust_list = []
        self.fear_list = []
        self.sad_list = []

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

    # 一定区間の瞬き回数をカウント
    def blink_true_count(self):
        cnt = 0
        for i in self.section_5_blink_list:
            if i == True:
                cnt += 1
        return cnt

    # 集中度を返す
    def get_concentration(self, move_num, max_frequency, min_frequency):
        try:
            return round((move_num - max_frequency) / (min_frequency - max_frequency), 2)
        except:
            return 0

    # weightを返す
    def get_weight(self, yaw, pitch, roll):
        return (1 - (((yaw / (self.angle_threshold_yaw * 10)) + (pitch / (self.angle_threshold_pitch * 10)) + (roll / (self.angle_threshold_roll * 10))) / 3))

    # 向きのyaw,pitch,rollを返す
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

    # 1フレーム前との差分を返す
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

    # earの計算結果を返す
    def calc_ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        eye_ear = (A + B) / (2.0 * C)
        return round(eye_ear, 3)

    # 瞬きをしたかの判定し，boolを返す関数
    def blink_count(self, left_eye_ear, right_eye_ear):
        blink_bool = False
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
                blink_bool = True

            self.COUNTER = 0
        return blink_bool

    # 顔の動きの差分とアングルの差分を返す
    def get_face_move(self, frame, rect, shape):
        x_diff = 0
        y_diff = 0
        pitch_diff = 0
        yaw_diff = 0
        roll_diff = 0

        # ロドリゲスのポイント抽出
        image_points = self.rodrigues_point(frame, shape)

        yaw, pitch, roll = self.get_rodrigues_angle(
            frame, image_points)
        print("yaw_:", yaw)
        print("pitch_:", pitch)
        print("roll_:", roll)
        if self.all_frame_cnt == 1:
            self.fast_yaw = abs(yaw)
            self.fast_pitch = abs(pitch)
            self.fast_roll = abs(roll)
            self.angle_threshold_up = self.fast_pitch - 12.5
            self.angle_threshold_down = self.fast_pitch + 12.5

        elif abs(yaw) < self.angle_threshold_yaw and abs(pitch) < self.angle_threshold_pitch and abs(roll) < self.angle_threshold_roll:
            print("yaw_if1: ", self.fast_yaw)
            print("pitch_if1: ", self.fast_pitch)
            print("roll_if1: ", self.fast_roll)
            print("yaw_if: ", yaw)
            print("pitch_if: ", pitch)
            print("roll_if: ", roll)
            yaw_diff = abs(abs(yaw) - abs(self.fast_yaw))
            pitch_diff = abs(abs(pitch) - abs(self.fast_pitch))
            roll_diff = abs(abs(roll) - abs(self.fast_roll))

        else:
            yaw_diff = self.angle_threshold_yaw
            pitch_diff = self.angle_threshold_pitch
            roll_diff = self.angle_threshold_roll

        # 動きの差分？
        print("yaw_deff", yaw_diff)
        print("roll_deff", roll_diff)
        print("pitch_deff", pitch_diff)
        x_diff, y_diff = self.get_move_diff(image_points)

        return x_diff, y_diff, yaw_diff, pitch_diff, roll_diff

    # 最終的な集中度の結果を返す
    def get_concentration_synthesis(self, c1, c2, w):
        return ((1 - w) * c1) + (w * c2)

    # 画像を保存する関数，返り値なし
    def save_image(self, frame):
        dt_now = datetime.datetime.now()
        self.full_save_file_path = self.save_dir_path + \
            "/" + str(dt_now) + ".jpg"
        cv2.imwrite(self.full_save_file_path, frame)

    # 画像を保存するファイルパスを設定，返り値なし
    def getFinalDirPath(self):
        root_dir_path = "../data/images/gc/"
        dir_list = os.listdir(root_dir_path)
        dir_list = [int(i) for i in dir_list]
        final_dir = 0
        if len(dir_list) != 0:
            dir_list = sorted(dir_list)
            final_dir = dir_list[-1]
        final_dir = root_dir_path + str(int(final_dir) + 1)
        os.mkdir(final_dir)
        self.save_dir_path = final_dir

    def create_dir(self):
        os.mkdir(self.save_dir_path)

    # 最大最低頻度を更新する関数
    def update_freq(self, blink, face_move):
        if self.max_blink_freq < blink:
            self.max_blink_freq = blink
        if self.min_blink_freq > blink or self.min_blink_freq == 0:
            self.min_blink_freq = blink
        if self.max_face_move_freq < face_move:
            self.max_face_move_freq = face_move
        if self.min_face_move_freq > face_move or self.min_face_move_freq == 0:
            self.min_face_move_freq = face_move

    def emotion_check(self, e):
        sad_sum = 0
        fear_sum = 0
        disgust_sum = 0
        angry_sum = 0

        if e != None:
            self.sad_list.append(e["Sad"])
            self.fear_list.append(e["Fear"])
            self.disgust_list.append(e["Disgust"])
            self.angry_list.append(e["Angry"])

            if len(self.sad_list) >= 20:
                self.sad_list.pop(0)
            if len(self.angry_list) >= 20:
                self.angry_list.pop(0)
            if len(self.disgust_list) >= 20:
                self.disgust_list.pop(0)
            if len(self.fear_list) >= 20:
                self.fear_list.pop(0)

            # if self.all_frame_cnt % 20 == 0:
            sad_sum = sum(self.sad_list)
            fear_sum = sum(self.fear_list)
            disgust_sum = sum(self.disgust_list)
            angry_sum = sum(self.angry_list)
            try:
                return (sad_sum / len(self.sad_list) + fear_sum / len(self.fear_list) +
                        disgust_sum / len(self.disgust_list) + angry_sum / len(self.angry_list)) / 4
            except:
                return 0

    # 動画からまとめた結果を返す
    def picture_read(self, picture_path):

        res_x = 0
        res_y = 0
        pitch_diff = 0
        yaw_diff = 0
        roll_diff = 0
        # res = {}
        c1 = 0
        c2 = 0
        c3 = 0
        blink_sum = 0
        face_move_sum = 0
        yaw_sum = 0
        pitch_sum = 0
        roll_sum = 0
        final_list = []
        w = 0
        files = os.listdir(picture_path)
        files.sort()
        for file in files:
            print(file)
            frame = cv2.imread(picture_path + "/" + file)

            # ランドマーク抽出
            rects = self.detector(frame, 0)

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

                self.section_5_blink_list.append(
                    self.blink_count(left_eye_ear, right_eye_ear))

                self.section_5_face_move_list.append(res_x + res_y)
                self.section_5_angle_list.append(
                    [yaw_diff, pitch_diff, roll_diff])

            # 顔を認識していないとき
            if (len(rects) == 0):
                self.section_5_blink_list.append(True)
                self.section_5_face_move_list.append(
                    self.max_face_move_freq / 10)
                self.section_5_angle_list.append(
                    [self.angle_threshold_yaw, self.angle_threshold_pitch, self.angle_threshold_roll])

            if (len(self.section_5_blink_list) > 10):
                self.section_5_blink_list.pop(0)
                self.section_5_face_move_list.pop(0)
                self.section_5_angle_list.pop(0)

            self.all_frame_cnt += 1

            if (self.all_frame_cnt >= 10):
                blink_sum = self.blink_true_count()
                face_move_sum = 0
                yaw_sum = 0
                pitch_sum = 0
                roll_sum = 0
                for i in self.section_5_face_move_list:
                    face_move_sum += i
                self.update_freq(blink_sum, face_move_sum)
                for i in self.section_5_angle_list:
                    yaw_sum += i[0]
                    pitch_sum += i[1]
                    roll_sum += i[2]
                print(type(float(blink_sum)))
                print(type(self.max_blink_freq))
                print(type(face_move_sum))
                c1 = self.get_concentration(
                    float(blink_sum), self.max_blink_freq, self.min_blink_freq)
                c2 = self.get_concentration(
                    float(face_move_sum), self.max_face_move_freq, self.min_face_move_freq)
                w = self.get_weight(yaw_sum, pitch_sum, roll_sum)
                c3 = self.get_concentration_synthesis(c1, c2, w)

            res = {
                "face_image_path": self.full_save_file_path,
                "time": file.replace(".jpg", ""),
                "blink": blink_sum,
                "face_move": face_move_sum,
                "angle": [yaw_sum, pitch_sum, roll_sum],
                "w": w,
                "c1": c1,
                "c2": c2,
                "c3": c3,
            }
            print("c2: ", c2)
            print("w: ", w)
            print("blink_count", self.blink_total)
            print(self.section_5_angle_list)
            final_list.append(res)

        return final_list

    # 動画からまとめた結果を返す
    def video_read(self, video_path):
        # hcd = HumanConditionDetection()
        final_list = []

        while True:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret != True:
                break
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            if self.all_frame_cnt == 100:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ランドマーク抽出
            rects = self.detector(frame, 0)
            res_x = 0
            res_y = 0
            pitch_diff = 0
            yaw_diff = 0
            roll_diff = 0
            # res = {}
            c1 = 0
            c2 = 0
            c3 = 0
            blink_sum = 0
            face_move_sum = 0
            yaw_sum = 0
            pitch_sum = 0
            roll_sum = 0
            emotion = None
            emotion_down = None
            # faces = self.face_cascade.detectMultiScale(
            #     frame, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
            w = 0
            for rect in rects:
                # try:
                # emotion = hcd.expression_estimation(frame, faces[0])
                # except:
                #     emotion = None
                # emotion_down = self.emotion_check(emotion)
                # try:
                #     emotion = hcd.expression_estimation(frame, faces[0])

                #     emotion_down = self.emotion_check(emotion)

                # except:
                #     emotion = None

                shape = self.predictor(frame, rect)
                shape = face_utils.shape_to_np(shape)

                # 顔の動きの座標とangleの差分を返す
                res_x, res_y, yaw_diff, pitch_diff, roll_diff = self.get_face_move(
                    frame, rect, shape)

                left_eye = shape[42:48]

                left_eye_ear = self.calc_ear(left_eye)

                right_eye = shape[36:42]

                right_eye_ear = self.calc_ear(right_eye)

                self.section_5_blink_list.append(
                    self.blink_count(left_eye_ear, right_eye_ear))

                self.section_5_face_move_list.append(res_x + res_y)
                self.section_5_angle_list.append(
                    [yaw_diff, pitch_diff, roll_diff])

            # 顔を認識していないとき
            if (len(rects) == 0):
                self.section_5_blink_list.append(True)
                self.section_5_face_move_list.append(
                    self.max_face_move_freq / 10)
                self.section_5_angle_list.append(
                    [self.angle_threshold_yaw, self.angle_threshold_pitch, self.angle_threshold_roll])

            if (len(self.section_5_blink_list) > 10):
                self.section_5_blink_list.pop(0)
                self.section_5_face_move_list.pop(0)
                self.section_5_angle_list.pop(0)
            self.all_frame_cnt += 1

            if (self.all_frame_cnt >= 10):
                # print(self.all_frame_cnt)
                # print(self.section_5_blink_list)

                face_move_sum = 0
                yaw_sum = 0
                pitch_sum = 0
                roll_sum = 0
                blink_sum = self.blink_true_count()
                for i in self.section_5_face_move_list:
                    face_move_sum += i
                self.update_freq(blink_sum, face_move_sum)
                for i in self.section_5_angle_list:
                    print(i)
                    yaw_sum += i[0]
                    pitch_sum += i[1]
                    roll_sum += i[2]
                c1 = self.get_concentration(
                    float(blink_sum), self.max_blink_freq, self.min_blink_freq)
                c2 = self.get_concentration(
                    float(face_move_sum), self.max_face_move_freq, self.min_face_move_freq)
                w = self.get_weight(yaw_sum, pitch_sum, roll_sum)
                c3 = self.get_concentration_synthesis(c1, c2, w)

            res = {
                "blink": blink_sum,
                "face_move": face_move_sum,
                "angle": [yaw_sum, pitch_sum, roll_sum],
                "w": w,
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "emotion": emotion,
                "emotion_down": emotion_down
            }
            final_list.append(res)

            # print("c2: ", c2)
            # print("w: ", w)
            # print("blink_count", self.blink_total)

        return final_list

    # websocketからまとめた結果を返す
    def main_face_landmark(self, base64_data):
        if(self.all_frame_cnt == 0):
            self.getFinalDirPath()
        self.all_frame_cnt += 1
        base64_data = base64_data[22:]
        base64_data = base64_data + '=' * (-len(base64_data) % 4)
        b = base64.urlsafe_b64decode(base64_data)
        img_numpy = np.fromstring(b, np.uint8)
        frame = cv2.imdecode(img_numpy, cv2.COLOR_BGR2GRAY)
        self.save_image(frame)

        # ランドマーク抽出
        rects = self.detector(frame, 0)
        res_x = 0
        res_y = 0
        pitch_diff = 0
        yaw_diff = 0
        roll_diff = 0
        # res = {}
        c1 = 0
        c2 = 0
        c3 = 0
        blink_sum = 0
        face_move_sum = 0
        yaw_sum = 0
        pitch_sum = 0
        roll_sum = 0

        w = 0
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

            self.section_5_blink_list.append(
                self.blink_count(left_eye_ear, right_eye_ear))

            self.section_5_face_move_list.append(res_x + res_y)
            self.section_5_angle_list.append([yaw_diff, pitch_diff, roll_diff])

        # 顔を認識していないとき
        if (len(rects) == 0):
            self.section_5_blink_list.append(True)
            self.section_5_face_move_list.append(self.max_face_move_freq / 10)
            self.section_5_angle_list.append(
                [self.angle_threshold_yaw, self.angle_threshold_pitch, self.angle_threshold_roll])

        if (len(self.section_5_blink_list) > 10):
            self.section_5_blink_list.pop(0)
            self.section_5_face_move_list.pop(0)
            self.section_5_angle_list.pop(0)

        self.all_frame_cnt += 1

        if (self.all_frame_cnt >= 10):
            blink_sum = self.blink_true_count()

            face_move_sum = 0
            yaw_sum = 0
            pitch_sum = 0
            roll_sum = 0
            for i in self.section_5_face_move_list:
                face_move_sum += i
            self.update_freq(blink_sum, face_move_sum)

            for i in self.section_5_angle_list:
                yaw_sum += i[0]
                pitch_sum += i[1]
                roll_sum += i[2]
            print(type(float(blink_sum)))
            print(type(self.max_blink_freq))
            print(type(face_move_sum))
            c1 = self.get_concentration(
                float(blink_sum), self.max_blink_freq, self.min_blink_freq)
            c2 = self.get_concentration(
                float(face_move_sum), self.max_face_move_freq, self.min_face_move_freq)
            w = self.get_weight(yaw_sum, pitch_sum, roll_sum)
            c3 = self.get_concentration_synthesis(c1, c2, w)

        res = {
            "face_image_path": self.full_save_file_path,
            "blink": blink_sum,
            "face_move": face_move_sum,
            "angle": [yaw_sum, pitch_sum, roll_sum],
            "w": w,
            "c1": c1,
            "c2": c2,
            "c3": c3,
        }
        print("c2: ", c2)
        print("w: ", w)
        print("blink_count", self.blink_total)
        print(self.section_5_angle_list)

        return res
