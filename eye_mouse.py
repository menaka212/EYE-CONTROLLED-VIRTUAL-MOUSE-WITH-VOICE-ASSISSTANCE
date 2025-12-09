import cv2
import time
import math
import pyautogui
import numpy as np
import pyttsx3

# ====== CONFIG ======
SHOW_PREVIEW = True
VOICE_ON = True
USE_DWELL_CLICK = True
DWELL_MS = 800
GAZE_SMOOTHING = 0.35
CURSOR_GAIN_X = 1.6
CURSOR_GAIN_Y = 1.4
BLINK_EAR_THRESH = 0.19
BLINK_MIN_FRAMES = 3
ROLL_SCROLL_THRESH = 8.0
SCROLL_STEP = 120
CAM_INDEX = 0

# ====== VOICE ======
engine = pyttsx3.init()
def speak(msg):
    if VOICE_ON:
        try:
            engine.say(msg)
            engine.runAndWait()
        except Exception:
            pass

# ====== HELPERS ======
def moving_avg(prev, new, alpha):
    if prev is None:
        return new
    return (1 - alpha) * new + alpha * prev

def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(pts):
    A = euclid(pts[1], pts[5])
    B = euclid(pts[2], pts[4])
    C = euclid(pts[0], pts[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

RIGHT_IRIS_IDX = [474, 475, 476, 477]
LEFT_IRIS_IDX  = [469, 470, 471, 472]

# ====== STATE ======
blink_counter = 0
blink_registered = False
last_click_time = 0.0
neutral_eye_center = None
smoothed_cursor = None
dwell_start_t = None
dwell_anchor = None

# Screen size
SCREEN_W, SCREEN_H = pyautogui.size()

def calibrate(neutral):
    global neutral_eye_center
    neutral_eye_center = neutral
    speak("Calibrated")

def normalized_to_screen(dx, dy):
    x = np.clip(SCREEN_W * (0.5 + dx * CURSOR_GAIN_X), 0, SCREEN_W-1)
    y = np.clip(SCREEN_H * (0.5 + dy * CURSOR_GAIN_Y), 0, SCREEN_H-1)
    return x, y

def roll_angle_deg(left_outer, right_outer):
    dx = right_outer[0] - left_outer[0]
    dy = right_outer[1] - left_outer[1]
    if abs(dx) < 1e-6:
        return 90.0 if dy > 0 else -90.0
    angle = math.degrees(math.atan2(dy, dx))
    return angle

# ====== MAIN ======
def main():
    global blink_counter, blink_registered, smoothed_cursor
    global dwell_start_t, dwell_anchor, neutral_eye_center
    global VOICE_ON, USE_DWELL_CLICK, last_click_time

    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    fps_t0 = time.time()
    frames = 0
    fps = 0.0

    speak("Eye control started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        cursor_target = None
        blink_now = False
        roll_now = 0.0

        if res.multi_face_landmarks:
            landmarks = res.multi_face_landmarks[0].landmark

            def lp(i):
                return (landmarks[i].x * w, landmarks[i].y * h)

            left_outer = lp(33)
            left_inner = lp(133)
            right_outer = lp(263)
            right_inner = lp(362)

            rpts = [lp(idx) for idx in RIGHT_EYE_IDX]
            lpts = [lp(idx) for idx in LEFT_EYE_IDX]
            ear_r = eye_aspect_ratio(rpts)
            ear_l = eye_aspect_ratio(lpts)
            ear = (ear_r + ear_l) * 0.5

            if ear < BLINK_EAR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_MIN_FRAMES:
                    blink_now = True
                blink_counter = 0

            r_iris = np.mean(np.array([lp(i) for i in RIGHT_IRIS_IDX]), axis=0)
            l_iris = np.mean(np.array([lp(i) for i in LEFT_IRIS_IDX]), axis=0)

            def norm_eye_center(outer, inner, iris):
                ex = (outer[0] + inner[0]) * 0.5
                ey = (outer[1] + inner[1]) * 0.5
                eye_w = max(abs(outer[0]-inner[0]), 1.0)
                eye_h = max(eye_w * 0.5, 1.0)
                nx = (iris[0] - ex) / eye_w
                ny = (iris[1] - ey) / eye_h
                return nx, ny

            nx_r, ny_r = norm_eye_center(right_outer, right_inner, r_iris)
            nx_l, ny_l = norm_eye_center(left_outer, left_inner, l_iris)
            nx = (nx_r + nx_l) * 0.5
            ny = (ny_r + ny_l) * 0.5

            if neutral_eye_center is None:
                neutral_eye_center = (nx, ny)

            dx = nx - neutral_eye_center[0]
            dy = ny - neutral_eye_center[1]

            target_x, target_y = normalized_to_screen(dx, dy)

            if smoothed_cursor is None:
                smoothed_cursor = np.array([target_x, target_y], dtype=float)
            else:
                smoothed_cursor = moving_avg(
                    smoothed_cursor,
                    np.array([target_x, target_y]),
                    GAZE_SMOOTHING
                )

            cursor_target = (int(smoothed_cursor[0]), int(smoothed_cursor[1]))

            roll_now = roll_angle_deg(left_outer, right_outer)
            if abs(roll_now) > ROLL_SCROLL_THRESH:
                pyautogui.scroll(SCROLL_STEP if roll_now < 0 else -SCROLL_STEP)

            if USE_DWELL_CLICK:
                if dwell_anchor is None:
                    dwell_anchor = cursor_target
                    dwell_start_t = time.time()
                else:
                    d = euclid(dwell_anchor, cursor_target)
                    if d < 25:
                        if (time.time() - dwell_start_t) * 1000 >= DWELL_MS:
                            pyautogui.click(cursor_target[0], cursor_target[1])
                            speak("Click")
                            dwell_start_t = time.time()
                    else:
                        dwell_anchor = cursor_target
                        dwell_start_t = time.time()

            now = time.time()
            if blink_now and (now - last_click_time) > 0.35:
                pyautogui.click(cursor_target[0], cursor_target[1])
                last_click_time = now
                speak("Click")

            if cursor_target:
                pyautogui.moveTo(cursor_target[0], cursor_target[1], duration=0)

            if SHOW_PREVIEW:
                for pt in [tuple(map(int, r_iris)), tuple(map(int, l_iris))]:
                    cv2.circle(frame, pt, 3, (0, 255, 0), -1)

                cv2.putText(frame, f"EAR:{ear:.3f} Roll:{roll_now:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)
        else:
            dwell_anchor = None
            dwell_start_t = None

        frames += 1
        if time.time() - fps_t0 >= 1.0:
            fps = frames / (time.time() - fps_t0)
            fps_t0 = time.time()
            frames = 0

        if SHOW_PREVIEW:
            htxt = [
                f"FPS:{fps:.1f}",
                "[C] Calibrate   [V] Voice   [D] Dwell",
                "[ESC/Q] Quit"
            ]
            y = 60
            for t in htxt:
                cv2.putText(frame, t, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 50), 2)
                y += 24
            cv2.imshow("Eye-Controlled Mouse", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break

        elif key in (ord('c'), ord('C')):
            neutral_eye_center = (nx, ny)
            speak("Calibrated")

        elif key in (ord('v'), ord('V')):
            VOICE_ON = not VOICE_ON
            speak("Voice on" if VOICE_ON else "Voice off")

        elif key in (ord('d'), ord('D')):
            USE_DWELL_CLICK = not USE_DWELL_CLICK
            speak("Dwell on" if USE_DWELL_CLICK else "Dwell off")

        elif key in (ord('r'), ord('R')):
            smoothed_cursor = None
            speak("Reset")

    cap.release()
    cv2.destroyAllWindows()
    speak("Stopped")

if __name__ == "__main__":
    main()
