import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import math

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="VisionMath AI",
    layout="wide"
)

st.title("VisionMath AI")
st.subheader("Draw Shapes in Air Using Green Tape on Finger")

st.write("""
Supported Shapes:
- Triangle
- Square
- Rectangle
- Pentagon
- Hexagon
- Heptagon
- Octagon
- Circle
- Ellipse
""")

st.info("Use BRIGHT GREEN tape/sticker on fingertip for best detection")

# ---------------- SESSION ----------------

if "clear" not in st.session_state:
    st.session_state.clear = False

if st.button("Clear Drawing"):
    st.session_state.clear = True

# ---------------- DETECTOR CLASS ----------------

class GeometryDetector(VideoTransformerBase):

    def __init__(self):
        self.points = []

    # Distance calculation
    def distance(self, p1, p2):

        return math.sqrt(
            (p1[0] - p2[0])**2 +
            (p1[1] - p2[1])**2
        )

    # Shape detection
    def detect_shape(self, approx):

        sides = len(approx)

        if sides == 3:
            return "Triangle"

        elif sides == 4:

            x, y, w, h = cv2.boundingRect(approx)

            ratio = w / float(h)

            if 0.95 <= ratio <= 1.05:
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

            (x, y), radius = cv2.minEnclosingCircle(approx)

            circle_area = math.pi * radius * radius

            ratio = area / circle_area

            if ratio > 0.75:
                return "Circle"

            return "Ellipse"

        return "Unknown"

    # Main transform
    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        if st.session_state.clear:
            self.points = []
            st.session_state.clear = False

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ---------------- GREEN DETECTION ----------------

        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        # Blur for smoothness
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.erode(mask, kernel)

        mask = cv2.dilate(mask, kernel)

        # ---------------- FIND CONTOURS ----------------

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

                # Smooth tracking
                if len(self.points) > 0:

                    prev_x, prev_y = self.points[-1]

                    smooth_x = int((prev_x + cx) / 2)
                    smooth_y = int((prev_y + cy) / 2)

                    self.points.append((smooth_x, smooth_y))

                else:
                    self.points.append((cx, cy))

                cv2.circle(
                    img,
                    (cx, cy),
                    10,
                    (0, 255, 0),
                    -1
                )

        # ---------------- DRAW TRACK PATH ----------------

        for p in self.points:

            cv2.circle(
                img,
                p,
                3,
                (255, 0, 0),
                -1
            )

        # ---------------- SHAPE ANALYSIS ----------------

        if len(self.points) > 40:

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

            shape = self.detect_shape(approx)

            cv2.polylines(
                img,
                [approx],
                True,
                (0, 255, 255),
                3
            )

            # ---------------- TEXT ----------------

            cv2.putText(
                img,
                f"Shape: {shape}",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            # ---------------- AREA ----------------

            area = cv2.contourArea(approx)

            cv2.putText(
                img,
                f"Area: {int(area)}",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # ---------------- PERIMETER ----------------

            perimeter = cv2.arcLength(
                approx,
                True
            )

            cv2.putText(
                img,
                f"Perimeter: {int(perimeter)}",
                (40, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # ---------------- SIDES ----------------

            sides = len(approx)

            cv2.putText(
                img,
                f"Sides: {sides}",
                (40, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # ---------------- CIRCLE DETAILS ----------------

            if shape == "Circle":

                (x, y), radius = cv2.minEnclosingCircle(approx)

                diameter = radius * 2

                cv2.putText(
                    img,
                    f"Radius: {int(radius)}",
                    (40, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    img,
                    f"Diameter: {int(diameter)}",
                    (40, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

        return img

# ---------------- START CAMERA ----------------

webrtc_streamer(
    key="visionmath-ai",
    video_transformer_factory=GeometryDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_transform=True
)
