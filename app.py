import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.title("AI Geometry Air Drawing")

mp_hands = mp.solutions.hands

class GeometryDetector(VideoTransformerBase):

    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.points = []

    def distance(self, p1, p2):
        return math.sqrt(
            (p1[0]-p2[0])**2 +
            (p1[1]-p2[1])**2
        )

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        h, w, c = img.shape

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                x = int(
                    hand_landmarks.landmark[8].x * w
                )

                y = int(
                    hand_landmarks.landmark[8].y * h
                )

                self.points.append((x, y))

                # Draw path
                for p in self.points:
                    cv2.circle(img, p, 5, (0,255,0), -1)

        # Detect shapes
        if len(self.points) > 50:

            pts = np.array(self.points, dtype=np.int32)

            epsilon = 0.02 * cv2.arcLength(pts, False)

            approx = cv2.approxPolyDP(
                pts,
                epsilon,
                False
            )

            sides = len(approx)

            cv2.polylines(
                img,
                [approx],
                False,
                (255,0,0),
                3
            )

            # Triangle
            if sides == 3:

                cv2.putText(
                    img,
                    "Triangle",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                a = self.distance(
                    approx[0][0],
                    approx[1][0]
                )

                b = self.distance(
                    approx[1][0],
                    approx[2][0]
                )

                c = self.distance(
                    approx[2][0],
                    approx[0][0]
                )

                perimeter = a+b+c

                s = perimeter/2

                area = math.sqrt(
                    s*(s-a)*(s-b)*(s-c)
                )

                cv2.putText(
                    img,
                    f"Area: {int(area)}",
                    (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    img,
                    f"Perimeter: {int(perimeter)}",
                    (50,140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

            # Rectangle or Square
            elif sides == 4:

                x1, y1 = approx[0][0]
                x2, y2 = approx[1][0]
                x3, y3 = approx[2][0]

                width = self.distance(
                    (x1,y1),
                    (x2,y2)
                )

                height = self.distance(
                    (x2,y2),
                    (x3,y3)
                )

                area = width * height

                perimeter = 2*(width+height)

                if abs(width-height) < 20:
                    shape = "Square"
                else:
                    shape = "Rectangle"

                cv2.putText(
                    img,
                    shape,
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                cv2.putText(
                    img,
                    f"Area: {int(area)}",
                    (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    img,
                    f"Perimeter: {int(perimeter)}",
                    (50,140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

            # Circle
            elif sides > 8:

                (cx, cy), radius = cv2.minEnclosingCircle(pts)

                area = math.pi * radius * radius

                circumference = 2 * math.pi * radius

                cv2.putText(
                    img,
                    "Circle",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                cv2.putText(
                    img,
                    f"Area: {int(area)}",
                    (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    img,
                    f"Circumference: {int(circumference)}",
                    (50,140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

        return img

webrtc_streamer(
    key="geometry",
    video_transformer_factory=GeometryDetector
)

st.markdown("Draw shapes in air using your index finger")
