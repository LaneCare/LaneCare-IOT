import requests
import json
from datetime import datetime

def upload_iot_report(base_url, userid, latitude, longitude, iot_id, image_file=None):
    """
    Uploads an IoT report to the specified API endpoint with an automatically generated description.

    Args:
    base_url (str): The base URL of the API (e.g., 'https://your-project.supabase.co/functions/v1/uploadData')
    userid (str): The user ID associated with the IoT device
    latitude (float): The latitude of the report location
    longitude (float): The longitude of the report location
    iot_id (str): The ID of the IoT device
    image_file (str, optional): Path to an image file to upload with the report

    Returns:
    dict: The JSON response from the API
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
        files['file'] = open(image_file, 'rb')

    try:
        # Make the POST request to the API
        response = requests.post(base_url, data=data, files=files)
        
        # Ensure we close the file if it was opened
        if image_file:
            files['file'].close()

        # Check if the request was successful
        response.raise_for_status()

        # Parse and return the JSON response
        return response.json()

    except requests.exceptions.RequestException as e:
        # Handle any errors that occurred during the request
        print(f"An error occurred: {e}")
        return {
            'status': 500,
            'message': 'Error occurred while making the request',
            'data': None
        }