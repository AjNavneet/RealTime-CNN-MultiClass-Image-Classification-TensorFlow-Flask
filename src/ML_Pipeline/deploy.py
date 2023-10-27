import tensorflow as tf
import numpy as np
from flask import Flask, request

# Import custom module Utils
import Utils

app = Flask(__name)

# Define the paths and variables for the API
model_path = '../output/cnn-model.h5'
input_image_path = "../output/api_input.jpg"
ml_model = Utils.load_model(model_path)
img_height = 180
img_width = 180
class_names = ['driving_license', 'others', 'social_security']

# Define an API endpoint for receiving images and getting their class predictions
@app.post("/get-image-class")
def get_image_class():
    # Receive the uploaded image from the request
    image = request.files['file']

    # Save the received image to a predefined input path
    image.save(input_image_path)

    # Load and preprocess the image for model input
    img = tf.keras.utils.load_img(input_image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions using the loaded model
    predictions = ml_model.predict(img_array)

    # Calculate class probabilities and choose the class with the highest probability
    score = tf.nn.softmax(predictions[0])
    output = {"class": class_names[np.argmax(score)], "confidence(%)": 100 * np.max(score)}

    return output

# Run the Flask app on host 0.0.0.0 and port 5001
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
