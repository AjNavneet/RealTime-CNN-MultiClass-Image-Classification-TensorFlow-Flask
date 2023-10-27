import pathlib
import subprocess

import tensorflow as tf

# Import custom modules from the ML_Pipeline package
from ML_Pipeline import Train_Model
from ML_Pipeline import Utils
from ML_Pipeline.Preprocess import apply
from ML_Pipeline.Utils import load_model, save_model

# Prompt the user to choose an action (0 for training, 1 for prediction, 2 for deployment)
val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))

if val == 0:  # Training
    # Define the data directory for training
    data_dir = pathlib.Path("../input/Training_data/")
    image_count = len(list(data_dir.glob('*/*')))
    print("Number of images for training: ", image_count)

    # Preprocess the data and obtain training and validation datasets along with class names
    train_ds, val_ds, class_names = apply(data_dir)

    # Train the machine learning model
    ml_model = Train_Model.fit(train_ds, val_ds, class_names)

    # Save the trained model to a specified path
    model_path = save_model(ml_model)
    print("Model saved in: ", model_path)

elif val == 1:  # Prediction
    # Specify the path to the pre-trained model
    model_path = "../output/cnn-model.h5"

    # Load the pre-trained model
    ml_model = load_model(model_path)

    # Define the directory containing test data
    test_data_dir = pathlib.Path("../input/Testing_Data/")
    image_count = len(list(test_data_dir.glob('*/*')))
    print("Number of images for testing: ", image_count)

    # Create a test dataset from the test data directory
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        seed=123,
        image_size=(Utils.img_height, Utils.img_width),
        batch_size=Utils.batch_size)

    # Make predictions using the loaded model
    prediction = ml_model.predict(test_ds)

    print(prediction)

    # Evaluate the model on the test dataset and print the results
    print(ml_model.evaluate(test_ds))

else:  # Deployment
    # For production deployment (uncomment this section for production)
    '''
    process = subprocess.Popen(['sh', 'ML_Pipeline/wsgi.sh'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )
    '''

    # For development deployment (comment out the production deployment section)
    process = subprocess.Popen(['python', 'ML_Pipeline/deploy.py'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )

    # Capture and print the standard output of the deployment process
    for stdout_line in process.stdout:
        print(stdout_line)

    # Wait for the process to finish and get the stdout and stderr
    stdout, stderr = process.communicate()
    print(stdout, stderr)
