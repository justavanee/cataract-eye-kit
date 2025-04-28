from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS, cross_origin
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the pre-trained model (replace with your actual model path)
model = load_model('model/cataract_model.keras')  # Update with your model path

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

    # Convert to numpy array and normalize (optional, based on how your model was trained)
    img_array = np.array(image) / 255.0  # Normalize if needed
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the result using the model
    prediction = model.predict(img_array)
    result = 'Cataract' if prediction[0][0] > 0.5 else 'Normal'
    
    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True, port=5000)