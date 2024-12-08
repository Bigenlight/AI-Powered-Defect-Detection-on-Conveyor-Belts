import time
import serial
import requests
import numpy as np
from io import BytesIO
from pprint import pprint
import cv2
import os
import random
from collections import Counter
import datetime

from requests.auth import HTTPBasicAuth
import json

# ==============================
# Parameters
# ==============================
CAPTURE_DELAY_FRAMES = 10  # 'data == b"0"' 신호 후 캡처까지 대기할 프레임 수
FREEZE_FRAMES = 30         # YOLO 결과를 표시할 프레임 수

# Expected counts
expected_counts = {
    'BOOTSEL': 1,
    'USB': 1,
    'CHIPSET': 1,
    'OSCILLATOR': 1,
    'RASPBERRY PICO': 1,
    'HOLE': 4
}

# Initialize serial communication with Arduino
ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1)

# Configuration for YOLO API
ACCESS_KEY = "ezeJWt9iFMaP7HGvwYgds6Za1Sb35fwHaPZF89mi"
AUTH_USERNAME = "kdt2024_1-27"
headers = {"Content-Type": "image/jpg"}
api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/8f81f503-b7c6-4220-8ad3-9e54ff2729c7/inference"

COLOR_LIST = [
    (255, 0, 0),      
    (50, 205, 0),     
    (0, 0, 255),      
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
        box = obj.get('box', [])
        if len(box) != 4:
            print(f"Invalid box format for object {obj}")
            continue
        try:
            x1, y1, x2, y2 = map(int, box)
        except ValueError:
            print(f"Non-integer box coordinates for object {obj}")
            continue

        color = get_color_for_class(cls, color_map)
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    y = 20
    (text_width, text_height), baseline = cv2.getTextSize('Text', font, font_scale, thickness)
    line_height = text_height + baseline + 5
    for label, count in label_counts.items():
        text = f"{label}: {count}"
        color = get_color_for_class(label, color_map)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(image, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), color, cv2.FILLED)
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
            print("Image sent successfully")
            print("Response JSON:")
            pprint(response.json())
            return response.json()
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
            print(f"Response content: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None

def draw_error_info(image, differences):
    """
    differences: list of tuples (class_name, difference)
    difference > 0: more
    difference < 0: less
    """

    # 이미지 테두리에 빨간색 라인
    cv2.rectangle(image, (0,0), (image.shape[1]-1, image.shape[0]-1), (0,0,255), 5)

    # 에러 메시지 표시 (좌하단)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_lines = []
    for cls, diff in differences:
        if diff > 0:
            text_lines.append(f"{cls} {diff} more")
        else:
            text_lines.append(f"{cls} {-diff} less")

    # 좌하단에서 위로
    # 밑에서부터 10픽셀 위에 첫 라인
    y = image.shape[0] - 10
    for line in reversed(text_lines):
        text = f"ERROR: {line}"
        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
        # text를 왼쪽 하단에 맞추기
        cv2.putText(image, text, (10, y), font, font_scale, (0,0,255), thickness, cv2.LINE_AA)
        y -= (th + 10)


def highlight_extra_objects(image, objects, differences):
    """
    differences: dict {class_name: difference}
    For classes with diff > 0 (more objects than expected), highlight the lowest scoring 'diff' objects with a semi-transparent yellow box.
    """
    # 필터링: difference가 양수인 클래스만 처리
    over_classes = {cls: diff for cls, diff in differences.items() if diff > 0}
    if not over_classes:
        return image

    # 객체를 클래스별로 모아서 점수순 정렬
    from operator import itemgetter
    class_objects = {}
    for obj in objects:
        cls = obj.get('class', 'N/A')
        if cls in over_classes:
            class_objects.setdefault(cls, []).append(obj)
    
    # 각 클래스별로 점수 낮은 순으로 정렬 후 diff 개수만큼 하이라이팅
    overlay = image.copy()
    for cls, diff in over_classes.items():
        objs = class_objects.get(cls, [])
        # 점수 낮은 순 정렬
        objs.sort(key=itemgetter('score'))
        # diff 개수만큼 반투명 노랑 박스
        highlight_objs = objs[:diff]
        yellow = (0,255,255)  # BGR
        for ho in highlight_objs:
            box = ho['box']
            x1, y1, x2, y2 = map(int, box)
            # 반투명 박스: overlay에 노랑 박스 그리고 원본과 합성
            cv2.rectangle(overlay, (x1, y1), (x2, y2), yellow, cv2.FILLED)
    
    # 반투명 합성 alpha=0.5
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

# 카메라 캡처
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera Error")
    exit(-1)

state = 'normal'
delay_count = 0
freeze_count = 0
annotated_image = None

crop_info = {"x": 870, "y": 110, "width": 600, "height": 530}
objects = []  # YOLO 결과 저장용 (freeze 상태에서 필요할 수 있음)

try:
    while True:
        if state == 'normal':
            ret, live_frame = cam.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            if crop_info is not None:
                live_frame = crop_img(live_frame, crop_info)

            if ser.in_waiting > 0:
                data = ser.read()
                print(f"Received data: {data}")
                if data == b"0":
                    state = 'pending'
                    delay_count = CAPTURE_DELAY_FRAMES
                    print(f"Switching to 'pending' state for {CAPTURE_DELAY_FRAMES} frames delay before capture.")

            cv2.imshow('Live', live_frame)

        elif state == 'pending':
            ret, frame = cam.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            if crop_info is not None:
                frame = crop_img(frame, crop_info)

            delay_count -= 1
            print(f"'pending' state: {delay_count} frames remaining before capture.")

            if delay_count <= 0:
                original_folder = 'original'
                if not os.path.exists(original_folder):
                    os.makedirs(original_folder)
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                original_image_path = os.path.join(original_folder, f"{timestamp}.jpg")
                cv2.imwrite(original_image_path, frame)
                print(f"Saved original image to {original_image_path}")

                result = inference_request(frame, api_url)
                label_counts = Counter()
                objects = []
                if result is not None:
                    print("YOLO Inference Result:")
                    pprint(result)
                    objects = result.get('objects', [])
                    if objects:
                        print(f"Number of objects detected: {len(objects)}")
                        for obj in objects:
                            cls = obj.get('class', 'N/A')
                            label_counts[cls] += 1

                        annotated_image = draw_bounding_boxes(frame.copy(), objects, color_map)
                        annotated_image = draw_label_counts(annotated_image, label_counts, color_map)

                        # 개수 비교
                        differences = {}
                        for cls, exp_count in expected_counts.items():
                            act_count = label_counts.get(cls, 0)
                            diff = act_count - exp_count
                            if diff != 0:
                                differences[cls] = diff

                        if differences:
                            # 에러 표시(빨간 테두리, 왼쪽 하단 ERROR)
                            draw_error_info(annotated_image, list(differences.items()))
                            # 초과된 객체들 반투명 노랑 박스
                            annotated_image = highlight_extra_objects(annotated_image, objects, differences)
                        print(f"annotated_image shape: {annotated_image.shape}")
                    else:
                        print("No objects detected.")
                        annotated_image = frame.copy()
                else:
                    print("Failed to get inference result.")
                    annotated_image = frame.copy()

                yolo_folder = 'Yolo'
                if not os.path.exists(yolo_folder):
                    os.makedirs(yolo_folder)
                annotated_image_path = os.path.join(yolo_folder, f"{timestamp}_annotated.jpg")
                cv2.imwrite(annotated_image_path, annotated_image)
                print(f"Annotated image saved to {annotated_image_path}")

                state = 'freeze'
                freeze_count = FREEZE_FRAMES
                # freeze 상태가 되자마자 Arduino에 b"1" 신호 전송
                ser.write(b"1")
                print(f"Switching to 'freeze' state for {FREEZE_FRAMES} frames and sent '1' to Arduino.")

        elif state == 'freeze':
            # freeze 상태에서 annotated_image 표시
            if annotated_image is not None:
                display_image = annotated_image.copy()
                max_width = 800
                max_height = 600
                height, width = display_image.shape[:2]
                if width > max_width or height > max_height:
                    scaling_factor = min(max_width / width, max_height / height)
                    display_image = cv2.resize(display_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

                cv2.imshow('Live', display_image)
                print("Displaying annotated image.")
            else:
                print("No annotated image to display.")

            freeze_count -= 1
            print(f"'freeze' state: {freeze_count} frames remaining.")

            # freeze 상태 동안 b"0" 수신 시 바로 pending으로 전환
            if ser.in_waiting > 0:
                data = ser.read()
                if data == b"0":
                    state = 'pending'
                    delay_count = CAPTURE_DELAY_FRAMES
                    print(f"Received b'0' in freeze state, switching to 'pending' state for {CAPTURE_DELAY_FRAMES} frames delay.")
                    continue

            # 표시 시간(대기시간) 조정
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                print("Exit key pressed. Exiting...")
                break

            if freeze_count <= 0:
                # freeze 끝나도 이제 여기서는 b"1" 안보냄 (이미 freeze 진입 시 보냈으므로)
                state = 'normal'
                print("Freeze ended, switching back to 'normal' state.")

        else:
            print("Unexpected state encountered.")
            break

        if state not in ('freeze',):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key pressed. Exiting...")
                break

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()
