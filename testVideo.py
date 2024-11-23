import os
import cv2
import threading
import requests
import supervision as sv
from inference import get_model
from datetime import datetime, timedelta

# API details
API_URL = "https://brwrlrfibkmgwxzgeugr.supabase.co/functions/v1/uploadData"
USER_ID = "db0372cf-f8e8-47c5-a547-08e86fb48437"
IOT_ID = "378168a4-5867-48e6-ae54-a625755e978b"

# Static GPS coordinates
STATIC_LATITUDE = 37.7749
STATIC_LONGITUDE = -122.4194

# Paths
CAPTURE_TEMP_PATH = os.path.join(os.getcwd(), "capture_temp")
if not os.path.exists(CAPTURE_TEMP_PATH):
    os.makedirs(CAPTURE_TEMP_PATH)

# Load the model (using CPU)
model = get_model(model_id="pothole-detection-project-bayaq/1", api_key="nWe7REaY8BsIq8grZ82f")

def upload_iot_report(base_url, userid, latitude, longitude, iot_id, image_file=None):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    description = f"Automated report from IoT device {iot_id} at {current_time}. Location: {latitude}, {longitude}"

    data = {
        'userid': userid,
        'latitude': str(latitude),
        'longitude': str(longitude),
        'description': description,
        'is_iot': 'true',
        'iot_id': iot_id
    }

    files = {}
    if image_file:
        files['file'] = ('image.jpg', image_file, 'image/jpeg')

    try:
        response = requests.post(base_url, data=data, files=files)
        response.raise_for_status()
        print(f"Response from API: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Upload error: {e}")

def save_image_temp(image, filename):
    file_path = os.path.join(CAPTURE_TEMP_PATH, filename)
    cv2.imwrite(file_path, image)
    return file_path

def delete_expired_files():
    current_time = datetime.now()
    for filename in os.listdir(CAPTURE_TEMP_PATH):
        file_path = os.path.join(CAPTURE_TEMP_PATH, filename)
        if os.path.isfile(file_path):
            file_creation_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if current_time - file_creation_time > timedelta(hours=1):
                os.remove(file_path)
                print(f"Deleted expired file: {file_path}")

# Video file input
VIDEO_PATH = "input_video.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

frame_counter = 0
fps = int(cap.get(cv2.CAP_PROP_FPS))
frames_to_skip = fps * 1  # Process every 3 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_counter += 1

    # Only process frames every 3 seconds
    if frame_counter % frames_to_skip != 0:
        continue

    # Run inference on the current frame
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)

    # If no potholes detected, continue to next frame
    if len(detections) == 0:
        continue

    # Annotate the frame with bounding boxes and labels
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imshow('Pothole Detection', annotated_frame)

    # Save and upload the frame if potholes are detected
    image_filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    file_path = save_image_temp(annotated_frame, image_filename)

    # Upload in a separate thread
    threading.Thread(target=upload_iot_report, args=(API_URL, USER_ID, STATIC_LATITUDE, STATIC_LONGITUDE, IOT_ID, open(file_path, 'rb'))).start()

    delete_expired_files()

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
