
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

     6 CNNs with ZeroPadding, comprised of 32, 64 and 128 filters of size 3X3.
     1 maxpool after every 2 CNNs and one dropout after the first 4 ones.
     3 FC layers

***** d) model is the best model and completed the track sucessfully

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

This model1 (getNvidiaModel) did not give satisfying results. The car was zigzagging left and right and eventually went off the road even before the first sharp turn.

My first thought was that the data was insufficent. I tried to remove some sekwed data but that did not help and the car had the same behavior.

Once I started normalizing the images for prediction, the results started to actually make sense. The car was steering gently
from left to right, staying in the middle of the road most of the time. The results were still not perfect though, as the
car went out of the road just before the end of the lap.

To prevent the car from steering away from the road,I decided to increase the steering correction on the left and right images.

The model was still not good enough. It seems that it was overfitting to the track and would sometimes turn too soon. I used more data augmentation to avoid this. I added some random brightness in my images.
Also, instead of using every image I have, I started to randomize the selection.

This seemed to help avoid overfitting and the car made a whole lap without falling. The steering angle was still changing too rapidly sometimes though.

To generate smoother data, I started going really slowly while capturing the data. This led to a way smoother variation in my angles and a way better data quality.

model4 worked best for me.

The car was driving way better now around the track.
