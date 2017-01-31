<<<<<<< HEAD
# Udacity Self-Driving Car Nanodegree
# Behavioral Cloning Project

Run the simulator with model

    a) python drive.py model.json
    b) Start the udacity provided simulator

How to create the model

    python model.py

Approach:

Data
----

    I used training data provided by Udacity. ./data directory contains
        driving_log.csv and IMG. I have not checked in IMG direcoty and available locally for model creation


Model creation
--------------

I have experimented with 4 differemt models


    a) model1 = getNvidiaModel()

        3 CNNs, comprised of 24,36,48,64, 64 filters of size 3X3
        4 Flatten/Dense laayer followed by Dropout and ELU
        Final Dense (1) layer

    b)  model2 = comma_model()

        3 CNNs, comprised of 16,32,64 filters of size 3X3
        2 Flatten followed by Dropout and ELU
        Final Dense layer


    c) model3 = getCnnModel2()

        3 CNNs, comprised of 32,16, 16 filters of size 3X3
        2 Dense layers 1024, 512 each followed by Dropout and ELU
        Final Dense layer


    d) model4 = getCnnModel1()

     6 CNNs, comprised of 32, 64 and 128 filters of size 3X3.
     1 maxpool after every 2 CNNs and one dropout after the first 4 ones.
     3 FC layers

** d) model is the best model and completed the track sucessfully

Main flow
---------

    read the data frame
    shuffle the data
    split data for training and validation
    initalize training_generator & validation_data_generator
    getModel
    train the model
    save the model


Testing and optimization
------------------------

This model1 did not give satisfying results.
The car was zigzagging left and right and eventually went off the road even before the first sharp turn.

My first thought was that the data was insufficent. I tried to remove some sekwed data but that did not help and the car had the same behavior.

Once I started normalizing the images for prediction, the results started to actually make sense. The car was steering gently
from left to right, staying in the middle of the road most of the time. The results were still not perfect though, as the
car went out of the road just before the end of the lap.

To prevent the car from steering away from the road,I decided to increase the steering correction on the left and right images.

The model was still not good enough. It seems that it was overfitting to the track and would sometimes turn too soon. I used more data augmentation to avoid this. I added some random brightness in my images.
Also, instead of using every image I have, I started to randomize the selection.

This seemed to help avoid overfitting and the car made a whole lap without falling. The steering angle was still changing too rapidly sometimes though.

To generate smoother data, I started going really slowly while capturing the data. This led to a way smoother variation in my angles and a way better data quality.

The car was driving way better now around the track.
=======
# About this project

This repository contains code for a project I did as a part of [Udacity's Self Driving Car Nano Degree Program](https://www.udacity.com/drive). We had to train a car to drive itself in a video game. The car was trained to drive itself using a deep neural network. Scroll down to see the demo video.

# Exploring the Dataset

I used the dataset provided by Udacity.

The dataset contains JPG images of dimensions 160x320x3. Here are some sample images from the dataset.

![Sample Images](assets/sample_images.png)

## Unbalanced Data
Most of the steering angles are close to zero, on the negative side. There is a bias towards driving straight and turning left.
![Unbalanced data](assets/unbalanced_data.png)

## Left/Right Camera ~= Parallel Transformation of the car
The left and right cameras point straight, along the length of the car. So the left and right camera are like parallel transformations of the car.
![Cameras](assets/cameras.png)

# Augmentation Techniques Used

I have to thank this [NVIDEA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and [this blog post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.d779iwp28) for suggesting these techniques.

## Use left & right camera images to simulate recovery
Using left and right camera images to simulate the effect of car wandering off to the side, and recovering. We will add a small angle .25 to the left camera and subtract a small angle of 0.25 from the right camera. The main idea being the left camera has to move right to get to center, and right camera has to move left.

## Flip the images horizontally
Since the dataset has a lot more images with the car turning left than right(because there are more left turns in the track), you can flip the image horizontally to simulate turing right and also reverse the corressponding steering angle.

## Brightness Adjustment
In this you adjust the brightness of the image to simulate driving in different lighting conditions

With these augmentation techniques, you can practically generate infinite unique images for training your neural network.

![Augmented Images](assets/augmentation.png)

# Preproceesing Images
1. I noticed that the hood of the car is visible in the lower portion of the image. We can remove this.
2. Also the portion above the horizon (where the road ends) can also be ignored.

After trial and error I figured out that shaving 55 pixels from the top and 25 pixels from the bottom works well.

Finally the image is resized to 64x64. Here is how a sample image could look like:

![**Final Image 64x64x3 image**](assets/resized.png)

## Data Generation Techniques Used
Data is augmented and generated on the fly using python generators. So for every epoch, the optimizer practically sees a new and augmented data set.

## Model Architecture

![Model Architecture](assets/model_architecture.png)

1. **Layer 1**: Conv layer with 32 5x5 filters, followed by ELU activation
2. **Layer 2**: Conv layer with 16 3x3 filters, ELU activation, Dropout(0.4) and 2x2 max pool
3. **Layer 3**: Conv layer with 16 3x3 filters, ELU activation, Dropout(0.4)
4. **Layer 4**: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation
5. **Layer 5**: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation

## Training Method

1. Optimizer: Adam Optimizer
2. No. of epochs: 3
3. Images generated per epoch: 20,000 images generated on the fly
3. Validation Set: 3000 images, generated on the fly
4. No test set used, since the success of the model is evaluated by how well it drives on the road and not by test set loss
5. Keras' `fit_generator` method is used to train images generated by the generator

## Evaluation Video



## How to run

python drive.py model.json


## How to run in autonomus mode (after training) and check the car driving properly

1. start the simulator (in your desktop, Applications/udacity-simulator)
2. python drive.py model.json

Click on the image to watch the video or [click here](https://youtu.be/kElUwEoZ7P0). You will be redirected to YouTube.

[![Demo Video](https://img.youtube.com/vi/kElUwEoZ7P0/0.jpg)](https://youtu.be/kElUwEoZ7P0)


>>>>>>> 9d5a6cbfb4067c68320b3a44b4012bc59ac2d37f
