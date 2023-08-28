import os
import sys
import platform
import random

import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from keras.layers import (BatchNormalization, Dense,
                          Dropout, GlobalAveragePooling2D)
from keras.regularizers import l2
from keras.models import (load_model as LoadModel, Model)
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.utils import plot_model

# Global Variables
image_width = 299
image_height = 299
batch_size = 16
num_epochs = 20


def main():
    '''
    Main function for Hotdog or Not Hotdog
    Program will train a model to classify images as hotdog or not hotdog
    If a model is provided as a command line argument, it will load the model and skip training
    It will then load a random image from the test directory and predict if it is a hotdog or not hotdog
    displaying the image and prediction label
    '''
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

        # Create the model
        model = getModel()

        # Generate a visualization of the model architecture
        plot_model(model, to_file='model_architecture.png',
                   show_shapes=True, show_layer_names=True)
        print("Model architecture visualization saved as 'model_architecture.png'")

        # Define the early stopping and model checkpoint callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = ModelCheckpoint(
            'hotdog_checkpoint.h5', save_best_only=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)

        # Train the model
        if os.path.exists('hotdog_checkpoint.h5'):
            # Load the weights from the checkpoint file
            print("Loading weights from 'hotdog_checkpoint.h5'")
            model.load_weights('hotdog_checkpoint.h5')

        # Train the model
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=num_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, lr_scheduler]
        )

        # Evaluate the model
        evaluation_results = model.evaluate(
            validation_generator, steps=len(validation_generator))
        print("Evaluation results:")
        print(f"Loss: {evaluation_results[0]}")
        print(f"Accuracy: {evaluation_results[1]}")

        # Save the final model
        model.save("hotdog.h5")
        print(f'Model saved to {os.getcwd()}/hotdog.h5')

    # Load Test Image (random from test directory which has images in test/hotdog and test/nothotdog) and Predict
    hotdog_dir = "dataset/test/hotdog"
    nothotdog_dir = "dataset/test/nothotdog"

    # Load all images from the directories
    hotdog_images = os.listdir(hotdog_dir)
    nothotdog_images = os.listdir(nothotdog_dir)

    # Create two separate lists for hotdog and not hotdog images
    hotdog_images_labeled = [(os.path.join(hotdog_dir, img), 1)
                             for img in hotdog_images]
    nothotdog_images_labeled = [
        (os.path.join(nothotdog_dir, img), 0) for img in nothotdog_images]

    # Combine the labeled images into a single list
    images = hotdog_images_labeled + nothotdog_images_labeled

    # Seed the random number generator
    random.seed()

    # Shuffle the images
    random.shuffle(images)

    # Initialize the index to 0
    index = 0

    while True:
        # Load the image
        test_image = load_img(
            images[index][0], target_size=(image_width, image_height))

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
        print(
            f'The image {images[index][0]} is a {prediction} with {result[0][0]} confidence')

        # Check if the prediction is correct
        correct_prediction = (prediction == 'hotdog' and images[index][1] == 1) or (
            prediction == 'not hotdog' and images[index][1] == 0)

        # Show the image with the prediction label
        showImagePrediction(images[index][0], prediction,
                            result[0][0], correct_prediction)

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
    '''
    returns a model with a VGG16 base model and custom classification layers
    '''
    # Load a pre-trained VGG16 model without the top classification layer
    base_model = tf.keras.applications.VGG16(
        weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers on top of the pre-trained model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    prediction = Dense(1, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=prediction)

    # Use the RMSprop optimizer with an initial learning rate
    initial_learning_rate = 0.001
    if platform.machine() in ['arm64', 'arm64e']:
        optimizer = tf.keras.optimizers.legacy.RMSprop(
            learning_rate=initial_learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=initial_learning_rate)

    # Compile the model with binary cross-entropy loss and accuracy metric
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def lr_schedule(epoch, initial_lr=0.001, min_lr=1e-6, max_lr=1e-3):
    '''
    Learning Rate Scheduler
    takes the epoch, initial learning rate, minimum learning rate, and maximum learning rate as arguments
    returns the learning rate for the epoch
    '''
    lr = initial_lr
    if epoch > 10:
        lr *= 0.1
    elif epoch > 5:
        lr *= 0.5
    lr = max(lr, min_lr)
    lr = min(lr, max_lr)
    return lr


def showImagePrediction(image_path, prediction, confidence, correct=True):
    '''
    Displays the image with the prediction label
    takes the image path, prediction label, confidence, and whether the prediction is correct as arguments
    '''
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

    cv2.putText(img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)

    # Show the image
    cv2.imshow("Image", img)


if __name__ == "__main__":
    main()
