import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import math

st.set_page_config(page_title="VisionMath AI")

st.title("VisionMath AI")
st.write("Draw shapes in air using BLACK tape on finger")

class GeometryDetector(VideoTransformerBase):

    def __init__(self):
        self.points = []

    def calculate_distance(self, p1, p2):
        return math.sqrt(
            (p1[0]-p2[0])**2 +
            (p1[1]-p2[1])**2
        )

    def detect_shape(self, approx):

        sides = len(approx)

        if sides == 3:
            return "Triangle"

        elif sides == 4:

            x, y, w, h = cv2.boundingRect(approx)

            aspect_ratio = float(w)/h

            if 0.95 <= aspect_ratio <= 1.05:
                return "Square"

            return "Rectangle"

        elif sides == 5:
            return "Pentagon"

        elif sides == 6:
            return "Hexagon"

        elif sides == 7:
            return "Heptagon"

        elif sides == 8:
            return "Octagon"

        elif sides > 8:

            area = cv2.contourArea(approx)

            (x,y), radius = cv2.minEnclosingCircle(approx)

            circle_area = math.pi * radius * radius

            ratio = area / circle_area

            if ratio > 0.75:
                return "Circle"

            return "Ellipse"

        return "Unknown"

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # BLACK color detection
        lower_black = np.array([0,0,0])
        upper_black = np.array([180,255,60])

        mask = cv2.inRange(
            hsv,
            lower_black,
            upper_black
        )

        kernel = np.ones((5,5), np.uint8)

        mask = cv2.erode(mask, kernel)

        mask = cv2.dilate(mask, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 1500:

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

        # Draw tracking path
        for p in self.points:

            cv2.circle(
                img,
                p,
                3,
                (255,0,0),
                -1
            )

        # Shape analysis
        if len(self.points) > 40:

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

            shape = self.detect_shape(approx)

            cv2.polylines(
                img,
                [approx],
                False,
                (0,255,255),
                3
            )

            cv2.putText(
                img,
                shape,
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                3
            )

            # PERIMETER
            perimeter = cv2.arcLength(
                approx,
                True
            )

            cv2.putText(
                img,
                f"Perimeter: {int(perimeter)}",
                (50,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            # AREA
            area = cv2.contourArea(approx)

            cv2.putText(
                img,
                f"Area: {int(area)}",
                (50,140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            # SIDES
            sides = len(approx)

            cv2.putText(
                img,
                f"Sides: {sides}",
                (50,180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            # Circle properties
            if shape == "Circle":

                (x,y), radius = cv2.minEnclosingCircle(approx)

                diameter = radius * 2

                cv2.putText(
                    img,
                    f"Radius: {int(radius)}",
                    (50,220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    img,
                    f"Diameter: {int(diameter)}",
                    (50,260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

        return img

webrtc_streamer(
    key="visionmath",
    video_transformer_factory=GeometryDetector
)
