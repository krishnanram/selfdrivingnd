import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from skimage.feature import hog
import pickle
import scipy.misc
from sklearn.model_selection import train_test_split

from Utils import *
from Explore import *


VEHICLES_DIR        = "./training_images/vehicles/"
NON_VEHICLES_DIR    = "./training_images/non-vehicles/"
IMAGES_DIR          = "./images/"
MODEL_DIR           = "./model/"
TEST_IMAGES_DIR     = "./testimages/"
OUTPUT_IMAGES       = "./output_images/"
HEATMAP             = "./heatmap/"

CELLPERBLOCK_LIST   = [2, 4]
ORIENT_LIST         = [9, 12]
HISTBINS_LIST       = [32, 48]

#CELLPERBLOCK_LIST   = [2]
#ORIENT_LIST         = [9]
#HISTBINS_LIST       = [48]

BEST_CELLPERBLOCK_LIST   = 2
BEST_ORIENT_LIST         = 9
BEST_HISTBINS_LIST       = 48

# In[8]:

'''
def displayHog(cars_train,notcars_train) :

    font_size = 15
    f, axarr = plt.subplots(4, 7, figsize=(20, 10))
    f.subplots_adjust(hspace=0.2, wspace=0.05)
    colorspace = cv2.COLOR_RGB2HLS
    # colorspace=cv2.COLOR_RGB2HSV
    # colorspace=cv2.COLOR_RGB2YCrCb

    i1, i2 = 22, 4000

    print(" Getting Hog features...")
    for ind, j in enumerate([i1, i2]):
        image = plt.imread(cars_train[j])
        feature_image = cv2.cvtColor(image, colorspace)

        axarr[ind, 0].imshow(image)
        axarr[ind, 0].set_xticks([])
        axarr[ind, 0].set_yticks([])
        title = "car {0}".format(j)
        axarr[ind, 0].set_title(title, fontsize=font_size)

        for channel in range(3):
            axarr[ind, channel + 1].imshow(feature_image[:, :, channel], cmap='gray')
            title = "ch {0}".format(channel)
            axarr[ind, channel + 1].set_title(title, fontsize=font_size)
            axarr[ind, channel + 1].set_xticks([])
            axarr[ind, channel + 1].set_yticks([])

        for channel in range(3):
            features, hog_image = get_hog_features(feature_image[:, :, channel], orient, pix_per_cell,
                                                   cell_per_block, vis=True, feature_vec=True)
            axarr[ind, channel + 4].imshow(hog_image, cmap='gray')
            title = "HOG ch {0}".format(channel)
            axarr[ind, channel + 4].set_title(title, fontsize=font_size)
            axarr[ind, channel + 4].set_xticks([])
            axarr[ind, channel + 4].set_yticks([])

    for indn, j in enumerate([i1, i2]):
        ind = indn + 2
        image = plt.imread(notcars_train[j])
        feature_image = cv2.cvtColor(image, colorspace)

        axarr[ind, 0].imshow(image)
        axarr[ind, 0].set_xticks([])
        axarr[ind, 0].set_yticks([])
        title = "not car {0}".format(j)
        axarr[ind, 0].set_title(title, fontsize=font_size)

        for channel in range(3):
            axarr[ind, channel + 1].imshow(feature_image[:, :, channel], cmap='gray')
            title = "ch {0}".format(channel)
            axarr[ind, channel + 1].set_title(title, fontsize=font_size)
            axarr[ind, channel + 1].set_xticks([])
            axarr[ind, channel + 1].set_yticks([])

        for channel in range(3):
            features, hog_image = get_hog_features(feature_image[:, :, channel], orient, pix_per_cell,
                                                   cell_per_block, vis=True, feature_vec=True)
            axarr[ind, channel + 4].imshow(hog_image, cmap='gray')
            title = "HOG ch {0}".format(channel)
            axarr[ind, channel + 4].set_title(title, fontsize=font_size)
            axarr[ind, channel + 4].set_xticks([])
            axarr[ind, channel + 4].set_yticks([])

    plt.show()
    # plt.savefig('./images/HOG_features_HLS.png')
    # plt.savefig('./images/HOG_features_YCrCb.png')


def displayMisclassifiedCarImage(cars_val,cars_val_feat,cars_nval,) :

    # plot false positives/negatives
    font_size = 15
    preds = svc.predict(cars_val_feat)
    misclassifieds = np.array(preds != np.ones(cars_nval))
    inds = np.where(preds != np.ones(cars_nval))
    inds = np.ravel(inds)
    misclassifieds = [cars_val[i] for i in inds]

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    fig.subplots_adjust(hspace=0.2, wspace=0.05)

    for i, ax in enumerate(axes.flat):
        print(i)
        # KRIS
        # ax.imshow(plt.imread(misclassifieds[i]))
        # xlabel = "false neg {0}".format(i)
        # ax.set_xlabel(xlabel)
        # ax.set_xticks([])
        # ax.set_yticks([])

    plt.show()
    print('number of misclassified car images', len(misclassifieds))
    # plt.savefig('./images/false_negatives.png')



def displayMisclassifiedNonCarImage(notcars_val_feat,ncars_nval,notcars_val) :

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    fig.subplots_adjust(hspace=0.2, wspace=0.05)

    preds = svc.predict(notcars_val_feat)
    inds = np.where(preds != np.zeros(ncars_nval))
    inds = np.ravel(inds)
    misclassifieds = [notcars_val[i] for i in inds]

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(misclassifieds[i]))
        xlabel = "false pos {0}".format(i)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    print('number of misclassified notcar images', len(misclassifieds))


# plt.savefig('./images/false_positives.png')

'''

class Train():

    def __init__(self):
        self.clf = None
        self.X_scaler = None

    # Read in the classifier and X_scaler
    def getModel(self):
        filename = MODEL_DIR+'model_' + str(BEST_ORIENT_LIST) + "_" + str(BEST_CELLPERBLOCK_LIST) + "_"+ str(BEST_HISTBINS_LIST) +'.pkl'
        pkl_file = open(filename, 'rb')
        self.clf, self.X_scaler = pickle.load(pkl_file)
        pkl_file.close()



    def extractFeatures(self,imgs, color_space='RGB', spatial_size=(32, 32),
                                         hist_bins=32, orient=9,
                                         pix_per_cell=8, cell_per_block=2,
                                         spatial_feat=True, hist_feat=True, hog_feat=True):

        print ("Inside extractFeatures")
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            # orig: image = mpimg.imread(file)

            image = scipy.misc.imread(file)

            # apply color conversion if other than 'RGB'
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                feature_image = np.copy(image)

            if spatial_feat == True:
                spatial_features = bin_spatial(feature_image, size=spatial_size)
                file_features.append(spatial_features)
            if hist_feat == True:
                # Apply color_hist()
                hist_features = color_hist(feature_image, nbins=hist_bins)
                file_features.append(hist_features)
            if hog_feat == True:
                # Extract every channel and append to features
                hog_features = get_hog_features(feature_image[:, :, 0], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                file_features.append(hog_features)

                hog_features = get_hog_features(feature_image[:, :, 1], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                file_features.append(hog_features)

                hog_features = get_hog_features(feature_image[:, :, 2], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                file_features.append(hog_features)

            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features

    def model(self):

        print ("Inside model creation ...")
        # Read data
        # Read in cars and notcars
        cars_list = glob.glob(VEHICLES_DIR+"*/*.png")
        not_cars_list = glob.glob(NON_VEHICLES_DIR +"*/*.png")

        cars = []
        notcars = []

        for car in cars_list:
            cars.append(car)

        for not_car in not_cars_list:
            notcars.append(not_car)

        color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        pix_per_cell = 8 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        spatial_size = (16, 16) # Spatial binning dimensions
        spatial_feat = True # Spatial features on or off
        hist_feat = True # Histogram features on or off
        hog_feat = True # HOG features on or off

        for cell_per_block in CELLPERBLOCK_LIST:
            for orient in ORIENT_LIST:
                for hist_bins in HISTBINS_LIST:

                    print(" Training with Cell per block {}. orient {}. hist_bins {}".format(cell_per_block, orient, hist_bins))

                    y_start_stop = [None, None] # Min and max in y to search in slide_window()
                    car_features = self.extractFeatures(cars, color_space=color_space, \
                                            spatial_size=spatial_size, hist_bins=hist_bins, \
                                            orient=orient, pix_per_cell=pix_per_cell, \
                                            cell_per_block=cell_per_block, \
                                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

                    notcar_features = self.extractFeatures(notcars, color_space=color_space, \
                                            spatial_size=spatial_size, hist_bins=hist_bins, \
                                            orient=orient, pix_per_cell=pix_per_cell, \
                                            cell_per_block=cell_per_block, \
                                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

                    X = np.vstack((car_features, notcar_features)).astype(np.float64)

                    # Fit a per-column scaler
                    X_scaler = StandardScaler().fit(X)

                    # Apply the scaler to X
                    scaled_X = X_scaler.transform(X)

                    # Define the labels vector
                    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

                    # Split up data into randomized training and test sets
                    rand_state = np.random.randint(0, 100)
                    X_train, X_test, y_train, y_test = train_test_split(
                        scaled_X, y, test_size=0.2, random_state=rand_state)

                    print('Using: ', orient, 'orientations', pix_per_cell,
                          'pixels per cell and ', cell_per_block, 'cells per block ',
                           hist_bins, 'hist_bins => feature vector length: ', len(X_train[0]))

                    # Use a linear SVC
                    svc = LinearSVC()
                    svc_model = CalibratedClassifierCV(svc)

                    # Check the training time for the SVC
                    t=time.time()
                    svc_model.fit(X_train, y_train)

                    t2 = time.time()
                    print('Seconds to train SVC: ' , round(t2-t, 2))
                    # Check the score of the SVC
                    print('Train Accuracy : ', round(svc_model.score(X_train, y_train), 4))
                    print('Test  Accuracy:  ', round(svc_model.score(X_test, y_test), 4))

                    # Check the prediction time for a single sample
                    i = rand_seed = np.random.randint(0, 1000)
                    t0 = time.time()
                    prediction = svc_model.predict(X_test[i].reshape(1, -1))
                    prob = svc_model.predict_proba(X_test[i].reshape(1, -1))

                    print("Prediction time : ", time.time()-t0)
                    print("Label ", int(y_test[i]), "Prediction {}".format(prediction))
                    print("Prob {}".format(prob))

                    # Save Model
                    filename = MODEL_DIR + 'model_' + str(orient) + '_' + str(cell_per_block) + "_" +  str(hist_bins) + '.pkl'
                    with open(filename, 'wb') as fp:
                        pickle.dump([svc_model, X_scaler], fp)

                    fp.close()


if __name__ == '__main__':

    print (" Training ...")
    train = Train()
    train.model()