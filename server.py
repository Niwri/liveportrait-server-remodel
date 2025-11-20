from flask import Flask, request, jsonify
from inference import InferencePipeline
from src.config.argument_config import ArgumentConfig
import uuid
import base64
from utils import *
import pickle

app = Flask(__name__)
inference_pipeline_map = dict()

@app.route('/test')
def test():
    return jsonify("Hello"), 200

@app.route('/register-connection')
def register_connection():
    global inference_pipeline_map

    id = uuid.uuid4()
    while id in inference_pipeline_map:
        id = uuid.uuid4()

    inference_pipeline_map[id] = InferencePipeline()
    print(f"Registered pipeline with ID {id}. Current number of pipelines: ", len(inference_pipeline_map.keys()))

    return jsonify({"id": id}), 200

@app.route('/remove-connection')
def remove_connection():
    global inference_pipeline_map

    data = request.get_json()
    if not data:
        return jsonify({"error": "Expected JSON body"}), 400

    id = data.get("id")
    if not id:
        return jsonify({"error": "Missing required field 'id'"}), 400

    id_uuid = None
    try:
        id_uuid = uuid.UUID(id)
    except Exception:
        return jsonify({"error": "Invalid UUID format"}), 400

    inference_pipeline_map.pop(id_uuid, None)

    return jsonify({"msg": f"Removed ID {id}"}), 200

# @app.route('/edit-face-image', methods=['POST'])
# def edit_face_image():
#     global inference_pipeline_map

#     data = request.get_json()
#     if not data:
#         return jsonify({"error": "Expected JSON body"}), 400

#     id = data.get("id")
#     if not id:
#         return jsonify({"error": "Missing required field 'id'"}), 400

#     id_uuid = None
#     try:
#         id_uuid = uuid.UUID(id)
#     except Exception:
#         return jsonify({"error": "Invalid UUID format"}), 400

#     if id_uuid not in inference_pipeline_map:
#         return jsonify({"error": f"ID {id_uuid} not found"}), 400

#     # Get image from data. Store it in local file named {input_(id).png}
#     source_img_b64 = data.get('source_image')
#     if not source_img_b64:
#         return jsonify({'error': 'No face_image provided'}), 400

#     source_img_bytes = None
#     source_img_path = INPUT_FOLDER + f'face_{id}.png'
#     try:
#         source_img_bytes = base64.b64decode(source_img_b64)
#         # Now you can save or process img_bytes as needed
#         # For example, save to file:
#         with open(source_img_path, 'wb') as f:
#             f.write(source_img_bytes)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

#     # Get source image from data. Store it in local file named {source_(id).png}
#     driving_img_b64 = data.get('driving_image')
#     if not driving_img_b64:
#         return jsonify({'error': 'No source_image provided'}), 400

#     driving_img_bytes = None
#     driving_img_path = INPUT_FOLDER + f'source_{id}.png'
#     try:
#         driving_img_bytes = base64.b64decode(driving_img_b64)
#         # Now you can save or process img_bytes as needed
#         # For example, save to file:
#         with open(driving_img_path, 'wb') as f:
#             f.write(driving_img_bytes)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

#     # Inference


#     # Output image in local file named {output_(id).png}
#     output_path = OUTPUT_FOLDER + f"output_{id}.png"

#     inference_pipeline_map[id_uuid].specify_source_path(source_img_path)
#     inference_pipeline_map[id_uuid].specify_driving_path(driving_img_path)
#     inference_pipeline_map[id_uuid].specify_output_path(output_path)

#     inference_pipeline_map[id_uuid].inference()

#     # Send image
#     output_img_b64 = None
#     with open(output_path, 'rb') as f:
#         output_img_bytes = f.read()
#         output_img_b64 = base64.b64encode(output_img_bytes).decode('utf-8')

#     return jsonify({'image': "data:image/jpeg;base64," + output_img_b64}), 200

@app.route('/edit-face-landmarks', methods=['POST'])
def edit_face_landmarks():
    global inference_pipeline_map

    data = request.get_json()
    if not data:
        return jsonify({"error": "Expected JSON body"}), 400

    id = data.get("id")
    if not id:
        return jsonify({"error": "Missing required field 'id'"}), 400

    id_uuid = None
    try:
        id_uuid = uuid.UUID(id)
    except Exception:
        return jsonify({"error": "Invalid UUID format"}), 400

    if id_uuid not in inference_pipeline_map:
        return jsonify({"error": f"ID {id_uuid} not found"}), 400

    # Check if source image is present
    if inference_pipeline_map[id_uuid].processed_source() is False:
        return jsonify({"error": f"Source has not yet been added"}), 400


    # Get source image from data. Store it in local file named {source_(id).png}
    driving_pkl_b64 = data.get('driving_pkl')
    if not driving_pkl_b64:
        return jsonify({'error': 'No driving_pkl provided'}), 400

    driving_pkl_bytes = None
    driving_pkl_path = INPUT_FOLDER + f'driving_{id}.pkl'
    try:
        driving_pkl_bytes = base64.b64decode(driving_pkl_b64)
        data = pickle.loads(driving_pkl_bytes)
        if('c_eyes_lst' not in data.keys() and 'c_d_eyes_lst' not in data.keys()):
            return jsonify({'error': 'Missing c_eyes_lst in pkl'}), 400

        if('c_lip_lst' not in data.keys() and 'c_d_lip_lst' not in data.keys()):
            return jsonify({'error': 'Missing c_lip_lst in pkl'}), 400

        if('motion' not in data.keys()):
            return jsonify({'error': 'Missing motion in pkl'}), 400

        with open(driving_pkl_path, 'wb') as f:
            pickle.dump(data, f)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Inference
    # Output image in local folder {(id)}
    output_folder = OUTPUT_FOLDER + f"{id}"

    inference_pipeline_map[id_uuid].specify_driving_path(driving_pkl_path)
    inference_pipeline_map[id_uuid].specify_output_path(output_folder)

    output_path = inference_pipeline_map[id_uuid].inference()

    # Send image
    output_img_b64 = None
    with open(output_path, 'rb') as f:
        output_img_bytes = f.read()
        output_img_b64 = base64.b64encode(output_img_bytes).decode('utf-8')

    return jsonify({'image': "data:image/jpeg;base64," + output_img_b64}), 200


@app.route('/process-source', methods=['POST'])
def process_source():
    global inference_pipeline_map

    data = request.get_json()
    if not data:
        return jsonify({"error": "Expected JSON body"}), 400

    id = data.get("id")
    if not id:
        return jsonify({"error": "Missing required field 'id'"}), 400

    id_uuid = None
    try:
        id_uuid = uuid.UUID(id)
    except Exception:
        return jsonify({"error": "Invalid UUID format"}), 400

    if id_uuid not in inference_pipeline_map:
        return jsonify({"error": f"ID {id_uuid} not found"}), 400

    # Get image from data. Store it in local file named {input_(id).png}
    source_img_b64 = data.get('source_image')
    if not source_img_b64:
        return jsonify({'error': 'No source_image provided'}), 400

    source_img_bytes = None
    source_img_path = INPUT_FOLDER + f'source_{id}.png'
    try:
        source_img_bytes = base64.b64decode(source_img_b64)
        # Now you can save or process img_bytes as needed
        # For example, save to file:
        with open(source_img_path, 'wb') as f:
            f.write(source_img_bytes)
        inference_pipeline_map[id_uuid].process_source(source_img_bytes)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


    return jsonify({'message': 'processed'}), 200

if __name__ == '__main__':
    app.run(debug=True)
