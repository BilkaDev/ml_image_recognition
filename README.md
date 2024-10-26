# Project Title:

Gender Recognition from Face Images Using Convolutional Neural Networks

I chose the Python programming language for the project.
Python offers us an extensive ecosystem of libraries such as Numpy, Pandas, TensorFlow and PyTorch, which help a lot in
working on and analyzing datasets.
Python also has extensive community support for building models for machine learning, which is helpful for
troubleshooting and project development.

# Libraries used in the project:

- TensorFlow and keras module: I used this library to build, train and evaluate a convolutional neural network (CNN),
  which is used for image classification.
- Matplotlib: I used it to visualize sample images from a data set.
- Numpy and Pandas: I used it to manage and manipulate the data.
- Jupyter Notebook: was used to organize code, documentation and visualize results in an
  interactive.
- Open CV: was used to load images.

# Description of the selected datasets

I chose Gender Classification Dataset to complete the project:
https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset
The dataset consists of cropped images of male and female faces. It is divided into a training and validation catalog.
The training dataset contains ~23000 images of each class, and the validation dataset contains ~5500 images of each
class.

# Data processing adapted to the chosen method.

- After loading the training and validation data, I set the size for all images to 80x80px to ensure uniformity of model
  input and to simplify the learning process.
- The images are scaled to the [0,1] range by dividing the pixel values by 255. to ensure that the input values are
  small and uniform.
- I added a sequence of image processing layers that randomly rotate the images horizontally and perform random
  rotations of 5% (0.05 radians). Data augmentation is a technique that increases the diversity of training data by
  making small modifications.
- I created a convolutional model containing three convolutional layers, each combined with a MaxPooling layer for
  dimensionality reduction then added Flatten layers (to change the feature maps to a one-dimensional vector), Dense (
  128 neurons) and finally added a Dense layer (1 neuron for final classification).
- The model is compiled with the Adam optimizer and learning rate set to 0.0001, the BinaryCrossentropy loss function (
  this is a loss function used in binary classification problems where the model predicts the probability that the data
  belongs to one of two classes ) and to evaluate the model's performance I used the accuracy metric (the ratio of the
  number of correct predictions to the total number of samples ). This setting prepares the model for the training
  process.

# Prepare teaching and testing datasets.
From the downloaded Gender Classification Dataset, I created new sets:
Test - It was divided into 2084 images of women's faces and 2574 of men's faces. It will be used for the final
evaluation of the model.
Validation - Contains 5000 images of each class. Used to tune model parameters and avoid overfitting. Allows monitoring
and adjusting model performance during training without affecting the test set.
Training - Contains 22,000 images of each class. Contains most of the available data and is used to train the model.

### sample images from the training set

![example_images](https://github.com/BilkaDev/ml_image_recognition/blob/main/docs/example_images.png)

### tag label for the images

1: women face
2: men face

# Demonstrate the operation of the algorithm with results

The model consists of three convolution layers, three pooling layers, one flatten layer and two dense layers.

![model_layers](http://url/to/img.png)

# Model training

![model training](https://github.com/BilkaDev/ml_image_recognition/blob/main/docs/model_layers.png)

Training was stopped early (early stopping) after the 24th epoch, with the restoration of the model weights from the
best epoch, which was the 19th epoch.

Model accuracy: The model achieved high accuracy on the training and validation set, reaching about 97% on the
validation set.

Model loss: The loss on the validation set decreased steadily, reaching its lowest value of about 0.0807 at the best
epoch (epoch 19).

# Evaluation of the model at the test set

Precision:0.9672259092330933,
Recall: 0.9630924463272095,
Accuracy: 0.9615715146064758

High precision (Precision): The model is very effective in classifying positive cases.
High Sensitivity (Recall): The model effectively detects actual positive cases and rarely misses them.
High accuracy (Accuracy): The model's effectiveness is very good, with a high percentage of correct predictions.

# Prediction comparison:

Conducted test on random photos of male and female faces. The model correctly predicted the gender in both photos:
For the photo of a man, the prediction was 0.9994, which unambiguously indicated a man.
For the photo of a woman, the prediction was 0.0185, which clearly indicated a woman.

# Conclusions

The model achieved high accuracy on the validation set (about 97%) and high precision (96.72%), sensitivity (96.31%) and
accuracy (96.16%) on the test set. These results prove that the created convolutional neural network performs well in
gender classification based on face images.
During training, an early stopping technique was used
stopping (early stopping) after 24 epochs, which helped select the model with the best overall performance (epoch 19).
The stability of the model was confirmed by
a systematic decrease in loss on the validation set and an increase in accuracy.
In order to improve the process of training the model and ensure its high accuracy and reliability, it is necessary to
conduct a thorough manual inspection of the datasets to ensure that the images are assigned to the correct categories (
“male” and “female”).
