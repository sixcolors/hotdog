## Hotdog or Not Hotdog Classifier

### Description:
The "Hotdog or Not Hotdog Classifier" project is an image classification application that uses deep learning to determine whether an image contains a hotdog or not. It's inspired by the famous "SeeFood" app from the TV show Silicon Valley. The project is built using Python and the TensorFlow framework. It employs a Convolutional Neural Network (CNN) architecture to learn and recognize patterns in images, making it capable of distinguishing between hotdog and non-hotdog images.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/vIci3C4JkL0/0.jpg)](https://www.youtube.com/watch?v=vIci3C4JkL0)

### Key Features:

1. Binary classification: The model classifies images as either "Hotdog" or "Not Hotdog."
2. Deep Learning Architecture: The project utilizes a CNN architecture with a InceptionV3 base model and custom classification layers for image feature extraction and classification.
3. Data Preprocessing: Images are preprocessed and augmented using TensorFlow's ImageDataGenerator to improve model robustness.
4. Regularization Techniques: The model uses L2 regularization and dropout to prevent overfitting and improve generalization performance.
5. Learning Rate Scheduling: The learning rate is scheduled to gradually decrease during training to improve performance.
6. Training and Evaluation: The model is trained using a labeled dataset and evaluated on a separate test dataset to measure its accuracy. Early stopping and model checkpointing are used to prevent overfitting and save the best model weights.
7. Model Loading and Prediction: The code includes a main function that loads a trained model (if provided) and predicts the class of a random image from the test directory.
8. Command Line Arguments: The code includes command line argument parsing to specify the path to a pre-trained model.
9. Code Documentation: The code includes comments and docstrings to explain the purpose and functionality of each section of code.
11. Model Saving: Trained models are saved in the Hierarchical Data Format (HDF5) format for future use and deployment.
12. Flexibility: The project includes customizable parameters such as image dimensions, batch size, and model architecture.

### Requirements:
- Python 3.11
- Anaconda or Miniconda
- TensorFlow (>=2.0, <3.0)
- NumPy (>=1.0, <2.0)
- Matplotlib (>=3.0, <4.0)
- OpenCV (>=4.0, <5.0)
- PyDot (>=1.0, <2.0)

Python packages can be installed using the `requirements.txt` file provided in the repository.

### Getting Started:

- Clone the repository to your local machine.
```bash
git clone https://github.com/sixcolors/hotdog.git
```
- Navigate to the project directory.
```bash
cd hotdog
```
- Grab the Dataset: 
    - Create a folder named "dataset" in the project directory.
    - Download the dataset (https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog) and extract the images
    - Move the "test" and "train" folders to the "dataset" folder.
- Create a virtual environment and activate it.
```bash
conda create -n hotdog_classifier python=3.11
conda activate hotdog_classifier
```
- Install the required dependencies listed in the requirements.txt file.
```bash
conda install --file requirements.txt
```
- Run the main script to train the model or use the optional command line argument to load a pre-trained model.
```bash
python main.py <path_to_model>
```
- Interact with the model using the OpenCV-based interface to test random images.
    - Press "d" to view the next image.
    - Press "a" to view the previous image.
    - Press any other key to exit.
- The image will be classified as "Hotdog" or "Not Hotdog."
    - The label will be displayed below the image with the confidence score.
    - The label will have a green background if the image is classified as "Hotdog" and a red background if it's classified as "Not Hotdog."
    - A checkmark will be displayed above the label if the prediction is correct and an "X" will be displayed if it's incorrect.

### Contributions:
Contributions to the project are welcome! You can contribute by adding new features, improving model accuracy, enhancing user interaction, or optimizing the codebase. Feel free to open issues, fork the repository, and submit pull requests to collaborate with the community.

### Acknowledgments:
This project is built on the foundation of deep learning, computer vision, and the TensorFlow library. It's a fun and educational demonstration of how machine learning can be applied to image classification tasks.

### Disclaimer:
This project is for educational purposes and entertainment, showcasing the capabilities of image classification using deep learning. It's not intended for production use and may not be perfect in classifying all images accurately.
