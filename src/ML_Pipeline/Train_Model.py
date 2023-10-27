from ML_Pipeline import Utils
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf

# Function to train an ML model
def train(model, train_ds, val_ds):
    epochs = 30

    # Train the model on the training dataset and validate on the validation dataset
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    return model

# Function to initiate the model and training data
def fit(train_ds, val_ds, class_names):
    num_classes = len(class_names)

    # Define data augmentation to increase model robustness
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", input_shape=(Utils.img_height, Utils.img_width, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # Create a Sequential model
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # Compile the model with the specified optimizer, loss function, and metrics
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Print a summary of the model architecture
    print(model.summary())

    # Train the model using the training and validation datasets
    model = train(model, train_ds, val_ds)

    return model
