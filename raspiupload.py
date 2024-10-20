import os
import cv2
import time
import requests
import supervision as sv
from inference import get_model
from datetime import datetime, timedelta

# API details
API_URL = "https://brwrlrfibkmgwxzgeugr.supabase.co/functions/v1/uploadData"
USER_ID = "db0372cf-f8e8-47c5-a547-08e86fb48437"
IOT_ID = "378168a4-5867-48e6-ae54-a625755e978b"
LATITUDE = 37.7749  # Static latitude
LONGITUDE = -122.4194  # Static longitude

CAPTURE_TEMP_PATH = os.path.join(os.getcwd(), "capture_temp")

def upload_iot_report(base_url, userid, latitude, longitude, iot_id, image_file=None):
    """
    Uploads an IoT report to the specified API endpoint with an automatically generated description.
    """
    # Generate automatic description
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    description = f"Automated report from IoT device {iot_id} at {current_time}. Location: {latitude}, {longitude}"

    # Prepare the data for the API request
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
        # Make the POST request to the API
        response = requests.post(base_url, data=data, files=files)
        
        # Check if the request was successful
        response.raise_for_status()

        # Return the JSON response
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {
            'status': 500,
            'message': 'Error occurred while making the request',
            'data': None
        }

def save_image_temp(image, filename):
    """
    Save the image to the capture_temp folder if the API request fails.
    """
    if not os.path.exists(CAPTURE_TEMP_PATH):
        os.makedirs(CAPTURE_TEMP_PATH)  # Creates folder if it doesn't exist

    file_path = os.path.join(CAPTURE_TEMP_PATH, filename)
    cv2.imwrite(file_path, image)

    return file_path

def retry_failed_uploads():
    """
    Retry uploading images that failed to upload before.
    """
    for filename in os.listdir(CAPTURE_TEMP_PATH):
        file_path = os.path.join(CAPTURE_TEMP_PATH, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as image_file:
                # Try to re-upload the image
                response = upload_iot_report(
                    base_url=API_URL,
                    userid=USER_ID,
                    latitude=LATITUDE,
                    longitude=LONGITUDE,
                    iot_id=IOT_ID,
                    image_file=image_file
                )
                # If upload successful, delete the image
                if response['status'] == 200:
                    os.remove(file_path)
                    print(f"Successfully re-uploaded and deleted: {file_path}")
                    print(f"Response from API: {response}")
                else:
                    print(f"Failed to re-upload: {file_path}")

def delete_expired_files():
    """
    Delete files that have been in the folder for more than 1 hour.
    """
    current_time = datetime.now()
    for filename in os.listdir(CAPTURE_TEMP_PATH):
        file_path = os.path.join(CAPTURE_TEMP_PATH, filename)
        if os.path.isfile(file_path):
            file_creation_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if current_time - file_creation_time > timedelta(hours=1):
                os.remove(file_path)
                print(f"Deleted expired file: {file_path}")

# Load a pre-trained YOLOv8n model
model = get_model(model_id="pothole-detection-project-bayaq/1", api_key="nWe7REaY8BsIq8grZ82f")

# Open a connection to the webcam (change 0 if you have another camera module)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    # Retry any failed uploads before processing new ones
    retry_failed_uploads()

    # Wait for 5 seconds before capturing the image
    time.sleep(5)

    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run inference on the current frame
    results = model.infer(frame)[0]

    # Load the results into the supervision Detections API
    detections = sv.Detections.from_inference(results)

    # If there is at least one pothole detected, proceed with uploading
    if len(detections) > 0:
        # Annotate the frame with inference results
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Display the annotated frame
        cv2.imshow('Webcam Detection', annotated_frame)

        # Save the image temporarily
        image_filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        # Try to send the image to the API
        with open(save_image_temp(annotated_frame, image_filename), 'rb') as image_file:
            response = upload_iot_report(
                base_url=API_URL,
                userid=USER_ID,
                latitude=LATITUDE,
                longitude=LONGITUDE,
                iot_id=IOT_ID,
                image_file=image_file
            )
        
        # If the upload fails, the image stays in capture_temp; it will be retried on the next loop iteration
        if response['status'] == 200:
            os.remove(os.path.join(CAPTURE_TEMP_PATH, image_filename))
            print(f"Successfully uploaded: {image_filename}")
            print(f"Response from API: {response}")

    # Delete expired files from the temp folder
    delete_expired_files()

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
