from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS, cross_origin
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64
import os
import gdown

app = Flask(__name__)
CORS(app)  # Enable CORS

def download_model_if_not_exists():
    model_path = 'model/cataract_model.keras'
    model_dir = 'model'
    file_id = '1Wmvu-Y0nvs-kq7Q0M69zQZqg0Rzzh2Vg'
    url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
        print("Model downloaded successfully!")

try:
    download_model_if_not_exists()
except Exception as exc:
    print(f"Model download failed: {exc}")

# Now load the pre-trained model
model = load_model('model/cataract_model.keras')  # Update with your model path
print("Model loaded successfully!")

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))  # Update with your image size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def serve_home():
    # Serve the index.html file from the web_app folder
    return send_from_directory(os.path.join(app.root_path, 'web_app'), 'index.html')

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory("web_app", filename)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    img_data = data['image']
    
    # Decode the base64 image data
    img_bytes = base64.b64decode(img_data)
    
    # Open image
    image = Image.open(io.BytesIO(img_bytes))

    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Resize the image to model expected size
    image = image.resize((224, 224))

    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the result using the model
    prediction = model.predict(img_array)
    result = 'Cataract' if prediction[0][0] > 0.5 else 'Normal'
    
    return jsonify({'result': result})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host="0.0.0.0", port=port)