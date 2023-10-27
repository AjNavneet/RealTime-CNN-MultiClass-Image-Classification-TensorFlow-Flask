import keras

# Define batch size and image dimensions
batch_size = 32
img_height = 180
img_width = 180

# Function to save a Keras model to a specified path
def save_model(model):
    model.save("../output/cnn-model.h5")
    return True

# Function to load a Keras model from a specified path
def load_model(model_path):
    model = None
    try:
        model = keras.models.load_model(model_path)
    except:
        print("Please enter the correct path")
        exit(0)

    return model
