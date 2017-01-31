import numpy as np
import json
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.layers import Dense, ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout
from keras.preprocessing.image import img_to_array, load_img
import cv2

from keras.optimizers import Adam
rows, cols, ch = 64, 64, 3

TARGET_SIZE = (64, 64)

IMAGE_HEIGHT_CROP = 108
IMAGE_WIDTH_CROP = 320
STEERING_ADJUSTMENT = 1
AUTONOMOUS_THROTTLE = .2

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
LR = 1e-5
OPTIMIZER = Adam(lr=LR)
LOSS = 'mse'
NB_EPOCH = 10
BATCH_SIZE = 128

img_width = 64
img_height = 64

batch_size = 125
epochs = 3

def brightenImage(image):

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def toTargetSize(image):
    return cv2.resize(image, TARGET_SIZE)


def resize(image):

    cropped_image = image[55:135, :, :]
    processed_image = toTargetSize(cropped_image)
    return processed_image


def preprocess(image):
    image = resize(image)
    image = image.astype(np.float32)

    #Normalize image
    image = image/255.0 - 0.5
    return image


def augment(row):

    steering = row['steering']

    # randomly choose the camera to take the image from
    camera = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left anf right cameras
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    image = load_img("data/" + row[camera].strip())
    image = img_to_array(image)

    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # flip the image and reverse the steering angle
        steering = -1*steering
        image = cv2.flip(image, 1)

    # Apply brightness augmentation
    image = brightenImage(image)

    # Crop, resize and normalize the image
    image = preprocess(image)
    return image, steering


def getDataGenerator(data_frame, batch_size=32):
    N = data_frame.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index, row in data_frame.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = augment(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch


def getCnnModel1():

    input_shape = (3, img_width, img_height)
    model = Sequential()
    # 2 CNNs blocks comprised of 32 filters of size 3x3.
    model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='elu'))
    # Maxpooling
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # 2 CNNs blocks comprised of 64 filters of size 3x3.
    model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    # Maxpooling + Dropout to avoid overfitting
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))    

    # 2 CNNs blocks comprised of 128 filters of size 3x3.
    model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
    model.add(Convolution2D(128, 3, 3, activation='elu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='elu'))
    # Last Maxpooling. We went from an image (64, 64, 3), to an array of shape (8, 8, 128)
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))    

    # Fully connected layers part.
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='elu'))
    # Dropout here to avoid overfitting
    model.add(Dropout(0.5))    
    model.add(Dense(64, activation='elu'))
    # Last Dropout to avoid overfitting
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='elu'))    
    model.add(Dense(1))

    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    return model


def getNvidiaModel(dropout=.25):
    print('NVIDIA Model...')
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - .5,
                     input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    # Subsample == stride
    # keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, border_mode='valid')
    model.add(Convolution2D(24, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv1'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(36, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv2'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(48, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv3'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, init='he_normal', activation='elu',
                            subsample=(1, 1), name='conv4'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, init='he_normal', activation='elu',
                            subsample=(1, 1), name='conv5'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    # We think NVIDIA has an error and actually meant the flatten == 1152, so no Dense 1164 layer
    # model.add(Dense(1164, init='he_normal', name="dense_1164", activation='elu'))
    model.add(Dense(100, init='he_normal', name="dense_100", activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, init='he_normal', name="dense_50", activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, init='he_normal', name="dense_10", activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, init='he_normal', name="dense_1"))
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    return model



def comma_model():
    print('Comma Model...')
    model = Sequential()
    # Color conversion
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
                     output_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    return model



def getCnnModel2():
    model = Sequential()
    # model.add(Lambda(preprocess_batch, input_shape=(160, 320, 3), output_shape=(64, 64, 3)))

    # layer 1 output shape is 32x32x32
    model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    # layer 2 output shape is 15x15x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # layer 3 output shape is 12x12x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))

    # Flatten the output
    model.add(Flatten())

    # layer 4
    model.add(Dense(1024))
    model.add(Dropout(.3))
    model.add(ELU())

    # layer 5
    model.add(Dense(512))
    model.add(ELU())

    # Finally a single output, since this is a regression problem
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

if __name__ == "__main__":


    BATCH_SIZE = 32

    data_frame = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3])

    # shuffle the data
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    # 80-20 training validation split
    training_split = 0.8

    num_rows_training = int(data_frame.shape[0]*training_split)

    training_data = data_frame.loc[0:num_rows_training-1]
    validation_data = data_frame.loc[num_rows_training:]

    # release the main data_frame from memory
    data_frame = None

    training_generator = getDataGenerator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = getDataGenerator(validation_data, batch_size=BATCH_SIZE)

    model = getCnnModel1()
    #model = comma_model()
    #model = getCnnModel2()
    #model = getNvidiaModel()

    samples_per_epoch = (40000/BATCH_SIZE)*BATCH_SIZE

    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=3, nb_val_samples=3000)

    print("Saving model weights and configuration file.")

    model.save_weights('model.h5')  # always save your weights after training or during training
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
