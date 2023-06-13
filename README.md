# Flower Classifier

The Flower Classifier is a AI Web App(simplified and design not completed) designed to classify pictures of flowers into five main classes: daisy, dandelion, roses, sunflowers, and tulips. This README file provides an overview of the model, its usage, and important information for developers and users. In this, I tried experimenting with the swish activation function and playing around with the model's hyperparameters before experimenting with Transfer Learning.

## Model Overview

The Flower Classifier model is built using deep learning techniques and trained on a dataset consisting of labeled flower images. It leverages the power of convolutional neural networks (CNNs) to learn and extract meaningful features from the input images. The model architecture is designed to capture the unique characteristics and patterns specific to each flower class, by modifying the MobileNetV2 model and using transfer learning techniques to adapt it to capture the flower features.

## Dataset

The model was trained on a diverse dataset I obtained from Kaagle that contains images of flowers belonging to the five main classes: daisy, dandelion, roses, sunflowers, and tulips. The training, validation, and testing sets were split to evaluate the performance of the model accurately.

## Usage

To use the Flower Classifier model, follow the steps below:

1. Install the required dependencies. The model was developed using Python and popular deep learning libraries such as TensorFlow and Keras. Make sure you have the necessary libraries installed before proceeding.

2. Load the trained model. The model weights and architecture are saved in a file, typically with the extension `.h5`. You can load the model using the appropriate function provided by your deep learning framework.

3. Preprocess the input image. The input image should be preprocessed to match the requirements of the model. This typically involves resizing the image to the input dimensions expected by the model and normalizing pixel values.

4. Classify the flower. Pass the preprocessed image through the loaded model to obtain the predicted class probabilities. The model will output a probability distribution across the five classes. The class with the highest probability can be considered as the predicted class for the given input image.

## Example Code

Here's an example code snippet demonstrating the usage of the Flower Classifier model:

```python
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('flower_classifier_model.h5')

# Preprocess the input image
image = Image.open('flower_image.jpg')
image = image.resize((224, 224))  # Resize to match the input dimensions expected by the model
image = image / 255.0  # Normalize pixel values

# Classify the flower
predictions = model.predict(tf.expand_dims(image, 0))
predicted_class = tf.argmax(predictions, axis=1)[0]

# Get the class label
class_labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
predicted_label = class_labels[predicted_class]

print('Predicted flower class:', predicted_label)
```

Ensure that you have the model file (`improved_flower_classifier_model.h5`) and the flower image file (`flower_image.jpg`) available in the appropriate paths.
UPDATE: Now there is a UI(web app with not so cool design lol) that you can upload your picture and using Flask API it can run it on model and output the flower's class.

## Model Performance

The model's performance can be evaluated using various metrics such as accuracy, precision, recall, and F1-score. During the training process, these metrics are computed on the validation set to monitor the model's progress and select the best performing model.

The reported performance metrics of the Flower Classifier model are as follows:

- Accuracy: 88.6%

Please note that these metrics are indicative of the model's performance on the test set during evaluation. The actual performance may vary depending on the quality and diversity of the input images.

## Conclusion

The Flower Classifier model is an effective tool for automatically classifying flower images into the five main classes: daisy, dandelion, roses, sunflowers, and tulips. By following the usage instructions outlined in this README, developers can easily incorporate the model into their applications or further improve it by fine-tuning on additional data. Happy classifying!
