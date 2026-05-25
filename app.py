import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.title("Air Shape Detector")

class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.points = []

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # GREEN COLOR RANGE
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 1000:

                x, y, w, h = cv2.boundingRect(cnt)

                cx = x + w // 2
                cy = y + h // 2

                self.points.append((cx, cy))

                cv2.circle(
                    img,
                    (cx, cy),
                    8,
                    (0, 255, 0),
                    -1
                )

        # DRAW TRACK
        for p in self.points:

            cv2.circle(
                img,
                p,
                3,
                (255, 0, 0),
                -1
            )

        # SHAPE DETECTION
        if len(self.points) > 30:

            pts = np.array(
                self.points,
                np.int32
            )

            peri = cv2.arcLength(
                pts,
                False
            )

            approx = cv2.approxPolyDP(
                pts,
                0.02 * peri,
                False
            )

            sides = len(approx)

            shape = "Unknown"

            if sides == 3:
                shape = "Triangle"

            elif sides == 4:
                shape = "Rectangle"

            elif sides > 8:
                shape = "Circle"

            cv2.putText(
                img,
                shape,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

        return img

webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)
