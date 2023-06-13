from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from flask import render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained model
model = tf.keras.models.load_model("improved_flower_classifier.h5")

# Define the class names
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    image_height = 224
    image_width = 224

    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    # Read the uploaded image
    image = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # Preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_height, image_width))
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)

    # Make the prediction
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]

    # Render the result template with the classification result
    return render_template('result.html', class_name=class_name)


if __name__ == '__main__':
    app.run(debug = True)
