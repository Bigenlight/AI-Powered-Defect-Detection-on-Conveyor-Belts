import time
import serial
import requests
import numpy as np
from pprint import pprint
import cv2
import os
import random
from collections import Counter
import datetime
from requests.auth import HTTPBasicAuth
import json
import gradio as gr

# ==============================
# Parameters
# ==============================
CAPTURE_DELAY_FRAMES = 2  # 'data == b"0"' 후 대기 프레임
FREEZE_FRAMES = 15         # YOLO 결과 표시 프레임 수
B0_DEBOUNCE_TIME = 0.3     # b"0" 신호 디바운싱 시간 (초)

expected_counts = {
    'BOOTSEL': 1,
    'USB': 1,
    'CHIPSET': 1,
    'OSCILLATOR': 1,
    'RASPBERRY PICO': 1,
    'HOLE': 4
}

class_thresholds = {
    'BOOTSEL': 0.95,
    'USB': 0.94,
    'CHIPSET': 0.90,
    'OSCILLATOR': 0.94,
    'RASPBERRY PICO': 0.97,
    'HOLE': 0.80
}

# 시리얼 포트 설정 (환경에 맞게 수정)
ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1)

# YOLO API 설정
ACCESS_KEY = "ezeJWt9iFMaP7HGvwYgds6Za1Sb35fwHaPZF89mi"
AUTH_USERNAME = "kdt2024_1-27"
api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/8c223a14-5aaa-40b4-ad75-b1b96ffb4ab3/inference"

COLOR_LIST = [
    (255, 0, 0),
    (50, 205, 0),
    (0, 0, 0),
    (225, 205, 0),
    (255, 0, 255),
    (128, 0, 0),
    (0, 128, 0),
    (50, 0, 128),
    (128, 128, 0),
]

color_map = {}

def get_color_for_class(cls, color_map):
    if cls not in color_map:
        if len(color_map) < len(COLOR_LIST):
            color_map[cls] = COLOR_LIST[len(color_map)]
        else:
            color_map[cls] = tuple(random.randint(0, 255) for _ in range(3))
    return color_map[cls]

def draw_bounding_boxes(image, objects, color_map):
    for obj in objects:
        cls = obj.get('class', 'N/A')
        score = obj.get('score', 0)
        valid = obj.get('valid', True)
        box = obj.get('box', [])
        if len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = map(int, box)
        except ValueError:
            continue

        if valid:
            color = get_color_for_class(cls, color_map)
        else:
            color = (0,0,255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{cls}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_bg_x1 = x1
        label_bg_y1 = y1 - text_height - baseline if y1 - text_height - baseline > 0 else y1 + text_height + baseline
        label_bg_x2 = x1 + text_width
        label_bg_y2 = y1
        label_bg_y1 = max(label_bg_y1, 0)
        label_bg_x2 = min(label_bg_x2, image.shape[1])
        cv2.rectangle(image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, thickness=cv2.FILLED)

        brightness = sum(color)
        text_color = (0, 0, 0) if brightness > 600 else (255, 255, 255)
        cv2.putText(image, label, (label_bg_x1, label_bg_y2 - baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return image

def draw_label_counts(image, label_counts, color_map):
    x = 10
    y = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize('Text', font, font_scale, thickness)
    line_height = text_height + baseline + 5

    for label, count in label_counts.items():
        text = f"{label}: {count}"
        color = get_color_for_class(label, color_map)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(image, (x - 5, y - th - 5), (x + tw + 5, y + 5), color, cv2.FILLED)
        brightness = sum(color)
        text_color = (0, 0, 0) if brightness > 600 else (255, 255, 255)
        cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_height
        if y > image.shape[0]:
            break
    return image

def crop_img(img, size_dict):
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    return img[y : y + h, x : x + w]

def inference_request(img: np.array, api_url: str):
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()
    try:
        response = requests.post(
            url=api_url,
            auth=HTTPBasicAuth(AUTH_USERNAME, ACCESS_KEY),
            headers={"Content-Type": "image/jpg"},
            data=img_bytes
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def draw_error_info(image, differences):
    cv2.rectangle(image, (0,0), (image.shape[1]-1, image.shape[0]-1), (0,0,255), 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_lines = []
    for cls, diff in differences:
        if diff > 0:
            text_lines.append(f"{cls} {diff} more")
        else:
            text_lines.append(f"{cls} {-diff} less")

    y = image.shape[0] - 10
    for line in reversed(text_lines):
        text = f"ERROR: {line}"
        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.putText(image, text, (10, y), font, font_scale, (0,0,255), thickness, cv2.LINE_AA)
        y -= (th + 10)

def highlight_extra_objects(image, objects, differences):
    from operator import itemgetter
    over_classes = {cls: diff for cls, diff in differences.items() if diff > 0}
    if not over_classes:
        return image
    overlay = image.copy()
    class_objects = {}
    for obj in objects:
        cls = obj.get('class', 'N/A')
        if cls in over_classes:
            class_objects.setdefault(cls, []).append(obj)
    for cls, diff in over_classes.items():
        objs = class_objects.get(cls, [])
        objs.sort(key=itemgetter('score'))
        highlight_objs = objs[:diff]
        yellow = (0,255,255)
        for ho in highlight_objs:
            box = ho['box']
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), yellow, cv2.FILLED)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=np.float32)

state = 'normal'
delay_count = 0
freeze_count = 0
annotated_image = None
objects = []
last_b0_time = 0.0
crop_info = {"x": 870, "y": 110, "width": 600, "height": 530}
stop_flag = True

def run_conveyor_system():
    global state, delay_count, freeze_count, annotated_image, objects, last_b0_time, stop_flag

    stop_flag = False

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Camera Error: Cannot open camera.")

    try:
        while not stop_flag:
            if state == 'normal':
                ret, live_frame = cam.read()
                if not ret:
                    break
                live_frame = cv2.filter2D(live_frame, -1, sharpening_kernel)
                if crop_info is not None:
                    live_frame = crop_img(live_frame, crop_info)

                if ser.in_waiting > 0:
                    data = ser.read()
                    if data == b"0":
                        now = time.time()
                        if now - last_b0_time > B0_DEBOUNCE_TIME:
                            last_b0_time = now
                            state = 'pending'
                            delay_count = CAPTURE_DELAY_FRAMES
                            time.sleep(0.1)
                yield live_frame

            elif state == 'pending':
                ret, frame = cam.read()
                if not ret:
                    break
                frame = cv2.filter2D(frame, -1, sharpening_kernel)
                if crop_info is not None:
                    frame = crop_img(frame, crop_info)

                delay_count -= 1
                if delay_count <= 0:
                    original_folder = 'original'
                    if not os.path.exists(original_folder):
                        os.makedirs(original_folder)
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    original_image_path = os.path.join(original_folder, f"{timestamp}.jpg")
                    cv2.imwrite(original_image_path, frame)

                    result = inference_request(frame, api_url)
                    label_counts = Counter()
                    objects = []
                    differences = {}
                    if result is not None:
                        raw_objects = result.get('objects', [])
                        for obj in raw_objects:
                            cls = obj.get('class', 'N/A')
                            score = obj.get('score', 0)
                            threshold = class_thresholds.get(cls, 0.5)
                            if score >= threshold:
                                obj['valid'] = True
                                label_counts[cls] += 1
                            else:
                                obj['valid'] = False
                            objects.append(obj)

                        annotated_image = frame.copy()
                        if objects:
                            annotated_image = draw_bounding_boxes(annotated_image, objects, color_map)
                            annotated_image = draw_label_counts(annotated_image, label_counts, color_map)
                            
                            for cls, exp_count in expected_counts.items():
                                act_count = label_counts.get(cls, 0)
                                diff = act_count - exp_count
                                if diff != 0:
                                    differences[cls] = diff
                            
                            if differences:
                                draw_error_info(annotated_image, list(differences.items()))
                                annotated_image = highlight_extra_objects(annotated_image, objects, differences)
                        else:
                            annotated_image = frame.copy()
                    else:
                        annotated_image = frame.copy()

                    if differences:
                        yolo_folder = 'Yolo_defects'
                    else:
                        yolo_folder = 'Yolo'
                    if not os.path.exists(yolo_folder):
                        os.makedirs(yolo_folder)
                    annotated_image_path = os.path.join(yolo_folder, f"{timestamp}_annotated.jpg")
                    cv2.imwrite(annotated_image_path, annotated_image)

                    state = 'freeze'
                    freeze_count = FREEZE_FRAMES
                    ser.write(b"1")
                    time.sleep(0.1)
                yield frame

            elif state == 'freeze':
                display_image = annotated_image.copy() if annotated_image is not None else None
                if display_image is not None:
                    max_width = 800
                    max_height = 600
                    height, width = display_image.shape[:2]
                    if width > max_width or height > max_height:
                        scaling_factor = min(max_width / width, max_height / height)
                        display_image = cv2.resize(display_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                    yield display_image

                freeze_count -= 1

                if ser.in_waiting > 0:
                    data = ser.read()
                    if data == b"0":
                        now = time.time()
                        if now - last_b0_time > B0_DEBOUNCE_TIME:
                            last_b0_time = now
                            state = 'pending'
                            delay_count = CAPTURE_DELAY_FRAMES
                            time.sleep(0.1)
                            continue

                if freeze_count <= 0:
                    state = 'normal'
                    time.sleep(0.1)

            else:
                break
    finally:
        cam.release()

def stop_system():
    global stop_flag
    stop_flag = True

with gr.Blocks() as demo:
    gr.Markdown("# Conveyor System Inspection\n")
    gr.Markdown("이 페이지에서 **Start** 버튼을 누르면 컨베이어 시스템을 작동시켜 실시간 영상과 YOLO 분석결과를 확인할 수 있습니다. **Stop** 버튼으로 종료할 수 있습니다.")
    gr.Markdown("**Start**를 누르기 전까지 시스템은 가동되지 않습니다.")

    with gr.Row():
        start_btn = gr.Button("Start")
        stop_btn = gr.Button("Stop")

    image_output = gr.Image(label="Real-Time Conveyor Feed", type="numpy", height=480)

    # stream=True 제거
    start_btn.click(fn=run_conveyor_system, inputs=[], outputs=image_output, api_name="start_conveyor", queue=True)
    stop_btn.click(fn=stop_system, inputs=[], outputs=[])

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
