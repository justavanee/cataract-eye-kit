# Start off by 'pyenv activate tf-env'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from keras.models import load_model

# Load your trained model
model = load_model('cataract_app/cataract_model.keras')

def load_and_predict_image(image_path, target_size=(224, 224)):
    # Load and display image
    img = image.load_img(image_path, target_size=target_size)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict (output is a value between 0 and 1)
    prediction = model.predict(img_array)[0][0]

    # Convert sigmoid output to class
    predicted_class = int(prediction < 0.5)  # 0: cataract, 1: normal
    confidence = 1 - prediction if predicted_class == 1 else prediction

    class_labels = {0: 'normal', 1: 'cataract'}
    predicted_label = class_labels[predicted_class]
    
    print(f"\nPredicted class: {predicted_label} ({confidence * 100:.2f}%)\n")
    return predicted_label

# Use the function you wrote
image_path = 'test/eye3.jpg'
load_and_predict_image(image_path)