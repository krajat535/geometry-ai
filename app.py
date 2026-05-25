import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import math

st.title("AI Geometry Air Drawing")

class GeometryDetector(VideoTransformerBase):

    def __init__(self):
        self.points = []

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detect RED object
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

        for p in self.points:

            cv2.circle(
                img,
                p,
                4,
                (255,0,0),
                -1
            )

        if len(self.points) > 30:

            pts = np.array(
                self.points,
                dtype=np.int32
            )

            peri = cv2.arcLength(pts, False)

            approx = cv2.approxPolyDP(
                pts,
                0.02 * peri,
                False
            )

            sides = len(approx)

            cv2.polylines(
                img,
                [approx],
                False,
                (0,255,255),
                3
            )

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

            elif sides == 4:

                cv2.putText(
                    img,
                    "Rectangle",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

            elif sides > 8:

                cv2.putText(
                    img,
                    "Circle",
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
    "Use RED tape on finger and draw shapes in air"
)
