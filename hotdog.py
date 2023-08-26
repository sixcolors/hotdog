import os
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model as LoadModel
import numpy as np
from keras.preprocessing import image
import cv2

# Global Variables
image_width = 299
image_height = 299
batch_size=16
num_epochs = 10
steps_per_epoch = 100
num_training_steps = 100

def main():
    
    # Check command line arguments
    if len(sys.argv) not in [1, 2]:
        print("Usage: python hotdog.py [model]")
        sys.exit(1)

    elif len(sys.argv) == 2:
        # Load the model
        model = LoadModel(sys.argv[1])

    else:
        # Part 1 - Data Preprocessing
        # Preprocessing the Training set
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=(image_width, image_height),
            batch_size=16,
            class_mode='binary',
            classes=['hotdog', 'nothotdog']
        )

        # Load the test set
        validation_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

        validation_generator = validation_datagen.flow_from_directory(
            'dataset/test_set',
            target_size=(image_width, image_height),
            batch_size=batch_size,
            class_mode='binary',
            classes=['hotdog', 'nothotdog']
        )

        model = getModel()

        # Train the model
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=num_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            batch_size=batch_size
        )

        # Evaluate the model
        evaluation_results = model.evaluate_generator(validation_generator, steps=len(validation_generator))
        print("Evaluation results:")
        print(f"Loss: {evaluation_results[0]}")
        print(f"Accuracy: {evaluation_results[1]}")

        # Save the model
        model.save("hotdog.h5")
        print(f'Model saved to {os.getcwd()}/hotdog.h5')

    # Load Test Image (random from test directory which has images in test/hotdog and test/nothotdog) and Predict
    hotdog_dir = "dataset/test_set/hotdog"
    nothotdog_dir = "dataset/test_set/nothotdog"

    # Load all images from the directories
    hotdog_images = os.listdir(hotdog_dir)
    nothotdog_images = os.listdir(nothotdog_dir)

    # Shuffle the images
    random.shuffle(hotdog_images)
    random.shuffle(nothotdog_images)

    # Combine the images into a single list
    images = hotdog_images + nothotdog_images

    # Initialize the index to 0
    index = 0

    while True:
        # Load the image
        test_image_path = images[index]
        if test_image_path in hotdog_images:
            img_dir = hotdog_dir
        else:
            img_dir = nothotdog_dir
        test_image = image.load_img(os.path.join(img_dir, test_image_path), target_size=(image_width, image_height))

        # Convert the image to a numpy array
        test_image = image.img_to_array(test_image)

        # Normalize the image
        test_image /= 255.0

        # Add a fourth dimension to the image (since Keras expects a list of images)
        test_image = np.expand_dims(test_image, axis=0)

        # Make a prediction
        result = model.predict(test_image)

        # Print Prediction
        if result[0][0] > 0.5:
            prediction = 'hotdog'
        else:
            prediction = 'not hotdog'
        print(f'The image {test_image_path} is a {prediction} with {result[0][0]} confidence')

        correct_prediction = (test_image_path in hotdog_images and prediction == 'hotdog') or (test_image_path in nothotdog_images and prediction == 'not hotdog')

        # Show the image (original jpg)
        showImagePrediction(os.path.join(img_dir, test_image_path), prediction, result[0][0], correct_prediction)

        # Wait for user input
        key = cv2.waitKey(0)

        # Move to the next or previous image based on user input
        if key == ord('a'):
            index = (index - 1) % len(images)
        elif key == ord('d'):
            index = (index + 1) % len(images)
        else:
            break


def getModel():
    # Model Definition
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(image_width, image_height, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=256, activation='leaky_relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(units=1, activation='sigmoid')
    ])

    # Model Compilation
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Function to show the image with prediction label
def showImagePrediction(image_path, prediction, confidence, correct=True):
    # Load the image
    img = cv2.imread(image_path)

    # Add the prediction label to the image
    label = f"{prediction} ({confidence:.2f})"
    if correct:
        font_color = (0, 255, 0)
    else:
        font_color = (0, 0, 255)
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)

    # Show the image
    cv2.imshow("Image", img)

if __name__ == "__main__":
    main()