import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import math
import av

st.title("AI Geometry Air Drawing")

class GeometryDetector(VideoTransformerBase):

    def __init__(self):
        self.points = []

    def detect_shape(self, pts):

        peri = cv2.arcLength(pts, True)

        approx = cv2.approxPolyDP(
            pts,
            0.02 * peri,
            True
        )

        return approx

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # RED color range
        lower_red = np.array([0,120,70])
        upper_red = np.array([10,255,255])

        mask = cv2.inRange(
            hsv,
            lower_red,
            upper_red
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 1000:

                x,y,w,h = cv2.boundingRect(cnt)

                cx = x + w//2
                cy = y + h//2

                self.points.append((cx,cy))

                cv2.circle(
                    img,
                    (cx,cy),
                    8,
                    (0,255,0),
                    -1
                )

        # Draw path
        for p in self.points:

            cv2.circle(
                img,
                p,
                4,
                (255,0,0),
                -1
            )

        # Shape detection
        if len(self.points) > 50:

            pts = np.array(
                self.points,
                dtype=np.int32
            )

            approx = self.detect_shape(pts)

            sides = len(approx)

            cv2.polylines(
                img,
                [approx],
                True,
                (0,255,255),
                3
            )

            # Circle
            if sides > 8:

                (x,y), radius = cv2.minEnclosingCircle(pts)

                area = math.pi * radius * radius

                perimeter = 2 * math.pi * radius

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
                    f"Circumference: {int(perimeter)}",
                    (50,140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

            # Triangle
            elif sides == 3:

                cv2.putText(
                    img,
                    "Triangle",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

            # Rectangle/Square
            elif sides == 4:

                cv2.putText(
                    img,
                    "Rectangle/Square",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

        return img

webrtc_streamer(
    key="geometry",
    video_transformer_factory=GeometryDetector
)

st.write(
    "Wear RED tape on finger and draw shapes in air"
)
