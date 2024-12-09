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

# 카메라 초기화
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera Error")
    # 실제 카메라 없을 경우 코드 실행이 안될 수 있음

# YOLO API 설정 (사용자 인증정보 변경 필요)
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

crop_info = {"x": 870, "y": 110, "width": 600, "height": 530}

sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=np.float32)

def get_color_for_class(cls, color_map):
    """클래스별 고유 색상 할당"""
    if cls not in color_map:
        if len(color_map) < len(COLOR_LIST):
            color_map[cls] = COLOR_LIST[len(color_map)]
        else:
            color_map[cls] = tuple(random.randint(0, 255) for _ in range(3))
    return color_map[cls]

def draw_bounding_boxes(image, objects, color_map):
    """바운딩 박스와 레이블 그리기"""
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
            # threshold 충족: 클래스별 색상
            color = get_color_for_class(cls, color_map)
        else:
            # threshold 미충족: 빨간색 표시
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
        cv2.rectangle(image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, cv2.FILLED)

        brightness = sum(color)
        text_color = (0, 0, 0) if brightness > 600 else (255, 255, 255)
        cv2.putText(image, label, (label_bg_x1, label_bg_y2 - baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return image

def draw_label_counts(image, label_counts, color_map):
    """레이블 개수를 이미지에 표시"""
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
    except requests.exceptions.RequestException as e:
        return None

def draw_error_info(image, differences):
    """에러 정보를 이미지에 그리기"""
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

def capture_and_infer():
    if not cam.isOpened():
        return "Camera not found.", None
    
    ret, frame = cam.read()
    if not ret:
        return "Failed to capture image", None

    # 샤프닝 적용
    frame = cv2.filter2D(frame, -1, sharpening_kernel)
    # 크롭 적용
    if crop_info is not None:
        frame = crop_img(frame, crop_info)

    # YOLO 추론
    result = inference_request(frame, api_url)
    if result is None:
        return "Inference failed.", None

    objects = []
    label_counts = Counter()
    differences = {}
    raw_objects = result.get('objects', [])

    # threshold 기반 valid 체크
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

    # OpenCV 이미지를 Gradio에 표시하기 위해 RGB 변환
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if differences:
        message = "Defects found!"
    else:
        message = "No defects!"

    return message, annotated_image

# Gradio UI 구성
with gr.Blocks() as demo:
    gr.Markdown("# PCB Inspection (YOLO Inference)\n카메라로 이미지를 캡처하고 YOLO로 검출한 결과를 시각화합니다.")
    with gr.Row():
        capture_button = gr.Button("Capture and Analyze")
    with gr.Row():
        result_message = gr.Textbox(label="Result Message")
    with gr.Row():
        annotated_image_output = gr.Image(label="Annotated Image")

    capture_button.click(fn=capture_and_infer, 
                         inputs=[],
                         outputs=[result_message, annotated_image_output])

# 페이지 실행
# 아래 코드 실행 시 로컬 주소 및 share=True로 공유 가능한 주소 모두 표시
demo.launch(share=True)
