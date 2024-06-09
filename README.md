# Hotdog or Not Hotdog Classifier

## Table of Contents
- [Description](#description)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Clean Up](#clean-up)
- [Contributions](#contributions)
- [Known Issues](#known-issues)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)
- [Disclaimer](#disclaimer)

## Description
Inspired by the "SeeFood" app from the TV show Silicon Valley, the "Hotdog or Not Hotdog Classifier" is an image classification application that uses deep learning to identify whether an image contains a hotdog. The project leverages Python and the TensorFlow framework, employing a Convolutional Neural Network (CNN) to learn and recognize patterns in images, thus distinguishing between hotdog and non-hotdog images.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/vIci3C4JkL0/0.jpg)](https://www.youtube.com/watch?v=vIci3C4JkL0)

## Key Features
1. **Binary Classification:** Images are classified as either "Hotdog" or "Not Hotdog."
2. **Deep Learning Architecture:** The project uses a CNN architecture with an InceptionV3 base model and custom classification layers for image feature extraction and classification.
3. **Data Preprocessing:** TensorFlow's ImageDataGenerator is used for image preprocessing and augmentation to enhance model robustness.
4. **Regularization Techniques:** L2 regularization and dropout are employed to prevent overfitting and improve model performance.
5. **Learning Rate Scheduling:** The learning rate is scheduled to decrease gradually during training to optimize performance.
6. **Training and Evaluation:** The model is trained on a labeled dataset and evaluated on a separate test dataset. Early stopping and model checkpointing are used to prevent overfitting and save the best model weights.
7. **Model Loading and Prediction:** The code includes a function that loads a trained model (if provided) and predicts the class of a random image from the test directory.
8. **Command Line Arguments:** The code includes command line argument parsing to specify the path to a pre-trained model.
9. **Code Documentation:** Comments and docstrings are included to explain the purpose and functionality of each section of code.
10. **Model Saving:** Trained models are saved in the Hierarchical Data Format (HDF5) for future use and deployment.
11. **Flexibility:** The project includes customizable parameters such as image dimensions, batch size, and model architecture.

## Requirements
- Python 3.11
- Anaconda or Miniconda
- TensorFlow (>=2.0, <3.0)
- NumPy (>=1.0, <2.0)
- Matplotlib (>=3.0, <4.0)
- OpenCV (>=4.0, <5.0)
- PyDot (>=1.0, <2.0)

Install the Python packages using the `requirements.txt` file provided in the repository.

## Getting Started
1. Clone the repository to your local machine.
```bash
git clone https://github.com/sixcolors/hotdog.git
```
2. Navigate to the project directory.
```bash
cd hotdog
```
3. Create a folder named "dataset" in the project directory, download the dataset from [here](https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog), extract the images, and move the "test" and "train" folders to the "dataset" folder.
4. Create a virtual environment and activate it.
```bash
conda create -n hotdog_classifier python=3.11
conda activate hotdog_classifier
```
5. Install the required dependencies listed in the requirements.txt file.
```bash
conda install --file requirements.txt
```

## Usage
1. Run the main script to train the model or use the optional command line argument to load a pre-trained model.
```bash
python main.py <path_to_model>
```
2. Interact with the model using the OpenCV-based interface to test random images.
    - Press "d" to view the next image.
    - Press "a" to view the previous image.
    - Press any other key to exit.
3. The image will be classified as "Hotdog" or "Not Hotdog."
    - The label will be displayed below the image with the confidence score.
    - The label will have a green background if the image is classified as "Hotdog" and a red background if it's classified as "Not Hotdog."
    - A checkmark will be displayed above the label if the prediction is correct and an "X" will be displayed if it's incorrect.

## Clean Up
Deactivate the virtual environment when you're done using the project.
```bash
conda deactivate
```

## Contributions
Contributions are welcome! You can contribute by adding new features, improving model accuracy, enhancing user interaction, or optimizing the codebase. Feel free to open issues, fork the repository, and submit pull requests to collaborate with the community.

## Known Issues
- No known issues at the moment.

## Future Improvements
- No improvements planned at the moment.

## Acknowledgments
This project is built on the foundation of deep learning, computer vision, and the TensorFlow library. It's a fun and educational demonstration of how machine learning can be applied to image classification tasks.

## Disclaimer
This project is for educational purposes and entertainment, showcasing the capabilities of image classification using deep learning. It's not intended for production use and may not be perfect in classifying all images accurately.