import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import math

st.title("Air Geometry AI")
st.write("Use GREEN tape on finger and draw shapes in air")

class ShapeDetector(VideoTransformerBase):

    def __init__(self):
        self.points = []

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # GREEN COLOR
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
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

        # DRAW PATH
        for p in self.points:

            cv2.circle(
                img,
                p,
                3,
                (255, 0, 0),
                -1
            )

        # DETECT SHAPE
        if len(self.points) > 30:

            pts = np.array(
                self.points,
                dtype=np.int32
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

                x, y, w, h = cv2.boundingRect(approx)

                ratio = w / float(h)

                if 0.95 <= ratio <= 1.05:
                    shape = "Square"
                else:
                    shape = "Rectangle"

            elif sides > 8:
                shape = "Circle"

            cv2.polylines(
                img,
                [approx],
                True,
                (0, 255, 255),
                3
            )

            # AREA
            area = cv2.contourArea(approx)

            # PERIMETER
            perimeter = cv2.arcLength(
                approx,
                True
            )

            cv2.putText(
                img,
                f"{shape}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            cv2.putText(
                img,
                f"Area: {int(area)}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Perimeter: {int(perimeter)}",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        return img

webrtc_streamer(
    key="geometry",
    video_transformer_factory=ShapeDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)
