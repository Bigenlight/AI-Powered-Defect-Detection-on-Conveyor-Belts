import time
import serial
import requests
import numpy as np
import cv2
import os
import random
from collections import Counter
import datetime
from requests.auth import HTTPBasicAuth
import json
import gradio as gr

CAPTURE_DELAY_FRAMES = 2
FREEZE_FRAMES = 15
B0_DEBOUNCE_TIME = 0.3

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

ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1)

ACCESS_KEY = "ezeJWt9iFMaP7HGvwYgds6Za1Sb35fwHaPZF89mi"
AUTH_USERNAME = "kdt2024_1-27"
api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/8c223a14-5aaa-40b4-ad75-b1b96ffb4ab3/inference"

crop_info = {"x": 870, "y": 110, "width": 600, "height": 530}

def crop_img(img, size_dict):
    if img is None or img.size == 0:
        print("crop_img: img is empty")
        return None
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    if y+h > img.shape[0] or x+w > img.shape[1]:
        print(f"crop_img: Cannot crop. img.shape={img.shape}, crop={size_dict}")
        return None
    return img[y : y + h, x : x + w]

stop_flag = True
state = 'normal'
delay_count = 0
freeze_count = 0
annotated_image = None
objects = []
last_b0_time = 0.0

def run_conveyor_system():
    global state, delay_count, freeze_count, annotated_image, objects, last_b0_time, stop_flag
    stop_flag = False

    # 카메라 해상도 설정 (카메라가 지원하는 해상도여야 함)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_FPS, 30)

    if not cam.isOpened():
        print("Camera Error: Cannot open camera.")
        raise RuntimeError("Camera Error: Cannot open camera.")

    print("Camera opened successfully.")

    try:
        while not stop_flag:
            if state == 'normal':
                ret, live_frame = cam.read()
                if not ret or live_frame is None or live_frame.size == 0:
                    print("No frame captured in normal state.")
                    break

                print(f"normal state: frame size = {live_frame.shape}")

                # 이미지 처리
                # live_frame = cv2.filter2D(live_frame, -1, sharpening_kernel)

                # 크롭 시도
                cropped = crop_img(live_frame, crop_info)
                if cropped is None or cropped.size == 0:
                    print("Cropping failed in normal state. Showing original frame without crop.")
                    # 크롭 실패시 우선 원본 프레임을 표시해 문제원인 확인
                    yield live_frame
                    continue
                else:
                    live_frame = cropped

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
                if not ret or frame is None or frame.size == 0:
                    print("No frame captured in pending state.")
                    break
                print(f"pending state: frame size = {frame.shape}")
                # frame = cv2.filter2D(frame, -1, sharpening_kernel)

                cropped = crop_img(frame, crop_info)
                if cropped is None or cropped.size == 0:
                    print("Cropping failed in pending state. Showing original frame without crop.")
                    yield frame
                    continue
                else:
                    frame = cropped

                delay_count -= 1
                if delay_count <= 0 and frame is not None and frame.size != 0:
                    # 실제 YOLO 추론 로직 여기서 수행 가능
                    pass

                yield frame

            elif state == 'freeze':
                if annotated_image is not None and annotated_image.size != 0:
                    yield annotated_image
                else:
                    print("freeze state: annotated_image is empty or None.")
                    # freeze 상태에서도 이미지 없으면 스킵
                    yield np.zeros((100,100,3), dtype=np.uint8) # 임시 빈 이미지 표시

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
                print("Unexpected state.")
                break
    finally:
        cam.release()
        print("Camera released.")

def stop_system():
    global stop_flag
    stop_flag = True
    print("Stop system called.")

with gr.Blocks() as demo:
    gr.Markdown("# Conveyor System Inspection\n")
    gr.Markdown("스타트 버튼을 누른 뒤 터미널에 표시되는 로그를 확인하여 카메라 프레임 크기와 크롭 가능 여부를 점검하세요.")

    with gr.Row():
        start_btn = gr.Button("Start")
        stop_btn = gr.Button("Stop")

    image_output = gr.Image(label="Real-Time Conveyor Feed (Cropped)", type="numpy", format="jpeg", height=480)

    start_btn.click(fn=run_conveyor_system, inputs=[], outputs=image_output, queue=True)
    stop_btn.click(fn=stop_system, inputs=[], outputs=[])

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
