import sys
import base64
import requests
from PIL import Image
from io import BytesIO
import pickle
import time

def send_edit_face_image(source_image_path, driving_pkl_path, id_value="testid"):
    # Read and encode images
    with open(source_image_path, 'rb') as f:
        source_b64 = base64.b64encode(f.read()).decode('utf-8')
    with open(driving_pkl_path, 'rb') as f:
        orgData = pickle.load(f)
        data = {
            'motion': orgData['motion'][orgData['n_frames'] // 2],
            'c_eyes_lst': orgData['c_eyes_lst'][orgData['n_frames'] // 2],
            'c_lip_lst': orgData['c_lip_lst'][orgData['n_frames'] // 2],
        }
        pkl_b64 = base64.b64encode(pickle.dumps(data)).decode('utf-8')

    source_payload = {
        "id": id_value,
        "source_image": source_b64,
    }
    url = "http://127.0.0.1:5000/process-source"
    startTime = time.time()
    resp = requests.post(url, json=source_payload)
    print("Processing source:", time.time() - startTime)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code}", resp.text)
        return

    payload = {
        "id": id_value,
        "driving_pkl": pkl_b64
    }
    url = "http://127.0.0.1:5000/edit-face-landmarks"
    startTime = time.time()
    resp = requests.post(url, json=payload)
    print("Inference:", time.time() - startTime)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code}", resp.text)
        return
    data = resp.json()
    img_data_uri = data.get("image")
    if not img_data_uri or not img_data_uri.startswith("data:image/jpeg;base64,"):
        print("No valid image in response.")
        return
    img_b64 = img_data_uri.split(",", 1)[1]
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(BytesIO(img_bytes))
    img.show()
    # input()


    # with open(driving_pkl_path, 'rb') as f:
    #     orgData = pickle.load(f)
    #     data = {
    #         'motion': orgData['motion'][orgData['n_frames'] - 1],
    #         'c_eyes_lst': orgData['c_eyes_lst'][orgData['n_frames'] - 1],
    #         'c_lip_lst': orgData['c_lip_lst'][orgData['n_frames'] - 1],
    #     }
    #     pkl_b64 = base64.b64encode(pickle.dumps(data)).decode('utf-8')
    # payload = {
    #     "id": id_value,
    #     "driving_pkl": pkl_b64
    # }
    # url = "http://127.0.0.1:5000/edit-face-landmarks"
    # startTime = time.time()
    # resp = requests.post(url, json=payload)
    # print("Inference:", time.time() - startTime)
    # if resp.status_code != 200:
    #     print(f"Error: {resp.status_code}", resp.text)
    #     return
    # data = resp.json()
    # img_data_uri = data.get("image")
    # if not img_data_uri or not img_data_uri.startswith("data:image/jpeg;base64,"):
    #     print("No valid image in response.")
    #     return
    # img_b64 = img_data_uri.split(",", 1)[1]
    # img_bytes = base64.b64decode(img_b64)
    # img = Image.open(BytesIO(img_bytes))
    # img.show()

def get_id():
    url = "http://127.0.0.1:5000/register-connection"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Error registering connection: {resp.status_code}", resp.text)
        return None
    data = resp.json()
    return data.get("id")


if __name__ == "__main__":
    print("Bruh")
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <face_image_path> <source_image_path>")
        sys.exit(1)
    id_value = get_id()
    if not id_value:
        print("Failed to get ID from server.")
        sys.exit(1)
    send_edit_face_image(sys.argv[1], sys.argv[2], id_value)
