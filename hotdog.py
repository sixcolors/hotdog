import os
import platform
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model as LoadModel
import numpy as np
from keras.preprocessing import image
import cv2

# Global Variables
image_width = 299
image_height = 299
batch_size=32
num_epochs = 16

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
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        train_generator = train_datagen.flow_from_directory(
            'dataset/train',
            target_size=(image_width, image_height),
            batch_size=batch_size,
            class_mode='binary',
            classes=['nothotdog', 'hotdog']
        )

        # Load the test set
        validation_datagen = ImageDataGenerator(rescale=1./255)

        validation_generator = validation_datagen.flow_from_directory(
            'dataset/test',
            target_size=(image_width, image_height),
            batch_size=batch_size,
            class_mode='binary',
            classes=['nothotdog', 'hotdog']
        )

        model = getModel()

        # Define the early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        # Train the model
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=num_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )

        # Evaluate the model
        evaluation_results = model.evaluate(validation_generator, steps=len(validation_generator))
        print("Evaluation results:")
        print(f"Loss: {evaluation_results[0]}")
        print(f"Accuracy: {evaluation_results[1]}")

        # Save the model
        model.save("hotdog.h5")
        print(f'Model saved to {os.getcwd()}/hotdog.h5')

    # Load Test Image (random from test directory which has images in test/hotdog and test/nothotdog) and Predict
    hotdog_dir = "dataset/test/hotdog"
    nothotdog_dir = "dataset/test/nothotdog"

    # Load all images from the directories
    hotdog_images = os.listdir(hotdog_dir)
    nothotdog_images = os.listdir(nothotdog_dir)

    # Create two separate lists for hotdog and not hotdog images
    hotdog_images_labeled = [(os.path.join(hotdog_dir, img), 1) for img in hotdog_images]
    nothotdog_images_labeled = [(os.path.join(nothotdog_dir, img), 0) for img in nothotdog_images]

    # Combine the labeled images into a single list
    images = hotdog_images_labeled + nothotdog_images_labeled

    # Shuffle the images
    random.shuffle(images)

    # Initialize the index to 0
    index = 0

    while True:
        # Load the image
        test_image = load_img(images[index][0], target_size=(image_width, image_height))

        # Convert the image to a numpy array
        test_image = img_to_array(test_image)

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
        print(f'Raw prediction: {result}')
        print(f'The image {images[index][0]} is a {prediction} with {result[0][0]} confidence')

        correct_prediction = (prediction == 'hotdog' and images[index][1] == 1) or (prediction == 'not hotdog' and images[index][1] == 0)

        # Show the image (original jpg)
        showImagePrediction(images[index][0], prediction, result[0][0], correct_prediction)

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
    # Load a pre-trained VGG16 model without the top classification layer
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))
    
    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification layers on top of the pre-trained model
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)  # Experiment with dropout rate
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)  # Experiment with dropout rate
    x = Dense(1, activation='sigmoid')(x)
    
    # Create the final model
    model = tf.keras.models.Model(base_model.input, x)
    
    # Use the Adam optimizer with a lower learning rate
    if platform.machine() in ['arm64', 'arm64e']:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # Compile the model with binary cross-entropy loss and accuracy metric
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Function to show the image with prediction label
def showImagePrediction(image_path, prediction, confidence, correct=True):
    # Load the image
    img = cv2.imread(image_path)

    # Add the prediction label to the image
    if prediction == 'hotdog':
        label = f"{prediction} ({confidence:.2f})"
    else:
        label = f"{prediction} ({1 - confidence:.2f})"

    # Set the font color based on whether the prediction is correct or not
    if correct:
        font_color = (0, 255, 0)
    else:
        font_color = (0, 0, 255)

    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)

    # Show the image
    cv2.imshow("Image", img)

if __name__ == "__main__":
    main()