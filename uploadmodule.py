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

# Static GPS coordinates
STATIC_LATITUDE = 37.7749  # Set your static latitude here
STATIC_LONGITUDE = -122.4194  # Set your static longitude here

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

        # Parse and return the JSON response
        json_response = response.json()
        print(f"Response from API: {json_response}")  # Print the response
        return json_response

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {
            'status': 500,
            'message': 'Error occurred while making the request',
            'data': None
        }

# def retry_failed_uploads():
#     """
#     Retry uploading images that failed to upload before.
#     """
#     for filename in os.listdir(CAPTURE_TEMP_PATH):
#         file_path = os.path.join(CAPTURE_TEMP_PATH, filename)
#         if os.path.isfile(file_path):
#             try:
#                 with open(file_path, 'rb') as image_file:
#                     # Try to re-upload the image using static GPS data
#                     response = upload_iot_report(
#                         base_url=API_URL,
#                         userid=USER_ID,
#                         latitude=STATIC_LATITUDE,
#                         longitude=STATIC_LONGITUDE,
#                         iot_id=IOT_ID,
#                         image_file=image_file
#                     )

#                 # If upload successful, delete the image
#                 if response['status'] == 200:
#                     os.remove(file_path)
#                     print(f"Successfully re-uploaded and deleted: {file_path}")
#                 else:
#                     print(f"Failed to re-upload: {file_path}")

#             except PermissionError as e:
#                 print(f"Permission error: {e}. File might still be in use: {file_path}")
#             except Exception as e:
#                 print(f"An error occurred while processing {file_path}: {e}")

#     return None  # Return None if no upload was successful

def save_image_temp(image, filename):
    """
    Save the image to the capture_temp folder if the API request fails.
    """
    if not os.path.exists(CAPTURE_TEMP_PATH):
        os.makedirs(CAPTURE_TEMP_PATH)  # Creates folder if it doesn't exist

    file_path = os.path.join(CAPTURE_TEMP_PATH, filename)
    cv2.imwrite(file_path, image)

    return file_path



# Load a pre-trained YOLOv8n model
model = get_model(model_id="pothole-detection-project-bayaq/1", api_key="nWe7REaY8BsIq8grZ82f")

# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    # # Retry any failed uploads before processing new ones
    # retry_failed_uploads()

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
                latitude=STATIC_LATITUDE,
                longitude=STATIC_LONGITUDE,
                iot_id=IOT_ID,
                image_file=image_file
            )
        
        # If the upload fails, the image stays in capture_temp; it will be retried on the next loop iteration
        if response['status'] == 200:
            os.remove(os.path.join(CAPTURE_TEMP_PATH, image_filename))
            print(f"Successfully uploaded: {image_filename}")

    # Delete expired files from the temp folder

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()


 # 1. Logging  (Report capture time)
 # 2. Check GPS location, save 2 last GPS data and compare with new capture, if GPS data same == reject, 
 # 3. Save image at file path /capture_temp then delete after 1 hour