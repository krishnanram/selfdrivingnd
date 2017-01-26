
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import tensorflow as tf
from tqdm import tqdm
from pylab import rcParams
from sklearn.metrics import confusion_matrix
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

plot=False
training_epochs = 100
log_batch_step = 100
batch_size = 250
test_batch_size = 250
test_batch_size = 10
img_shape = (32, 32, 3)


training_epochs = 1
log_batch_step = 100
batch_size = 1
test_batch_size = 1

def plot_probabilities(pred_cls, pred_prob, title) :

    plt.plot(list(pred_cls), list(pred_prob), 'ro')
    x1,x2,y1,y2 = plt.axes()
    plt.axes((x1-1, x2+1, y1, y2+0.1))
    plt.ylabel("Probability")
    plt.title(title)
    plt.show()





def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(np.argmax(cls_true[i]))
        else:
            xlabel = "True: {0}, Pred: {1}".format(np.argmax(cls_true[i]), np.argmax(cls_pred[i]))
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def plot_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=test_cls,
                          y_pred=cls_pred)

    plt.figure(figsize=(40,40))
    rcParams['figure.figsize'] = 13, 13
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, range(n_classes))
    plt.yticks(tick_marks, range(n_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_example_errors(cls_pred, correct):

    incorrect = (correct == False)
    images = test_features[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = test_cls[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def print_test_accuracy(session,test_features,oh_test_labels,test_cls,y_pred_cls,show_example_errors=False, show_confusion_matrix=False):

    num_test = len(test_features)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:

        j = min(i + test_batch_size, num_test)

        batch_features = test_features[i:j]
        batch_labels   = oh_test_labels[i:j]

        feed_dict = {input_ph: batch_features, labels_ph: batch_labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    correct = (test_cls == cls_pred)
    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        #plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        #plot_confusion_matrix(cls_pred=cls_pred)

def load() :

    print('Inside load() ...')
    # Load pickled data
    import pickle

    training_file = './data/train.p'
    testing_file = './data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    train_features, train_labels = train['features'], train['labels']
    test_features, test_labels = test['features'], test['labels']

    print('Done loading data')

    return  train_features, train_labels, test_features, test_labels

def dataSum(train_features, test_features) :

    print('Inside dataSum()...')

    n_train = len(train_features)
    n_test = len(test_features)
    image_shape = "{}x{}".format(len(train_features[0]), len(train_features[0][0]))
    n_classes = max(train_labels) + 1

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    return n_classes

def dataExploration(train_features, train_labels, test_features, test_labels) :

    print('Inside dataExploration()...')

    train_features = np.array(train_features)
    train_labels = np.array( train_labels)

    inputs_per_class = np.bincount(train_labels)
    max_inputs = np.max(inputs_per_class)

    if plot :

        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)
        ax.set_ylabel('Inputs')
        ax.set_xlabel('Class')
        ax.set_title('Number of inputs per class')
        ax.bar(range(len(inputs_per_class)), inputs_per_class, 1 / 3, color='blue', label='Inputs per class')
        plt.show()

        for i in range(n_classes):
            for j in range(len(train_labels)):
                if (i == train_labels[j]):
                    print('Class: ', i)
                    plt.imshow(train_features[j])
                    plt.show()
                    break



    return (inputs_per_class,max_inputs)

def preProcess(inputs_per_class,max_inputs,train_features,train_labels) :
    print('Inside preProcess() ...')
    # Generate additional data for underrepresented classes
    print('Generating additional data...')
    angles = [-5, 5, -10, 10, -15, 15, -20, 20, -25, 25]

    for i in range(len(inputs_per_class)):
        input_ratio = min(int(max_inputs / inputs_per_class[i]) - 1, len(angles) - 1)

        if input_ratio <= 1:
            continue

        new_features = []
        new_labels = []
        mask = np.where(train_labels == i)

        for j in range(input_ratio):
            for feature in train_features[mask]:
                new_features.append(scipy.ndimage.rotate(feature, angles[j], reshape=False))
                new_labels.append(i)

        train_features = np.append(train_features, new_features, axis=0)
        train_labels = np.append(train_labels, new_labels, axis=0)

    # Normalize features
    print('Normalizing features...')

    train_features = train_features / 255. * 0.8 + 0.1

    # Get 20% of training data as validation data

    train_features, valid_features, train_labels, valid_labels = train_test_split(
        train_features,
        train_labels,
        test_size=0.2,
        random_state=832289
    )
    if plot :
        inputs_per_class = np.bincount(train_labels)
        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)
        ax.set_ylabel('Inputs')
        ax.set_xlabel('Class')
        ax.set_title('Number of inputs per class')
        ax.bar(range(len(inputs_per_class)), inputs_per_class, 1 / 3, color='green', label='Inputs per class')
        plt.show()

    return train_features, valid_features, train_labels, valid_labels

def createNeuralNet(train_features, valid_features, train_labels, valid_labels) :

    print('Inside createNeuralNet()...')

    # Input dimensions
    image_width = len(train_features[0][0])
    image_height = len(train_features[0])
    color_channels = len(train_features[0][0][0])

    # Convolutional layer patch dinmension and output size
    # filter is based on 3*3 window
    filter_width = 3
    filter_height = 3

    #128 for k is reasonable
    conv_k_output = 128

    # Dimension parameters for each fully connected layer
    # It follows reduced dimension for first to second and 3 to fourth

    # List to hold fully connected layer dimension details.
    # With this technique, we can add more layers dynamically without chaning the code

    fc_params = [
        image_width * image_height * conv_k_output,
        1024,
        1024,
        n_classes
    ]

    # Build weights and biases
    conv2d_weight = None
    conv2d_bias = None
    fc_weights = []
    fc_biases = []

    with tf.variable_scope('BONHOMME', reuse=False):

        conv2d_weight = tf.get_variable("conv2w", shape=[filter_width, filter_height, color_channels, conv_k_output],
                                        initializer=tf.contrib.layers.xavier_initializer())

        conv2d_bias   = tf.get_variable("conv2b", shape=[conv_k_output],
                                      initializer=tf.contrib.layers.xavier_initializer())

        for i in range(len(fc_params) - 1):

            fc_weights.append(tf.get_variable('fc_weight' + str(i), shape=[fc_params[i], fc_params[i + 1]],
                                              initializer=tf.contrib.layers.xavier_initializer()))

            fc_biases.append(tf.get_variable('fc_bias' + str(i), shape=[fc_params[i + 1]],
                                             initializer=tf.contrib.layers.xavier_initializer()))

    # hot encoded training and validation labels
    oh_train_labels = tf.one_hot(train_labels, n_classes).eval(session=tf.Session())
    oh_valid_labels = tf.one_hot(valid_labels, n_classes).eval(session=tf.Session())

    # Input placeholders
    input_ph = tf.placeholder(tf.float32, shape=[None, image_width, image_height, color_channels])
    labels_ph = tf.placeholder(tf.float32)

    # Convolutional layer
    network = tf.nn.conv2d(input_ph, conv2d_weight, strides=[1, 1, 1, 1], padding='SAME')
    network = tf.nn.bias_add(network, conv2d_bias)
    network = tf.nn.relu(network)

    # Fully connected convolutional layers
    # read from the list ...

    for i in range(len(fc_weights)):

        network = tf.matmul(tf.contrib.layers.flatten(network), fc_weights[i]) + fc_biases[i]
        if i < len(fc_weights) - 1:  # No relu after last FC layer
            network = tf.nn.relu(network)

    # Loss computation, cross entropy for the last network (output..)
    prediction = tf.nn.softmax(network)
    cross_entropy = -tf.reduce_sum(labels_ph * tf.log(prediction + 1e-6), reduction_indices=1)
    loss = tf.reduce_mean(cross_entropy)

    # Accuracy computation
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_ph, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

    return input_ph, labels_ph, oh_train_labels, oh_valid_labels, loss, accuracy,prediction

def plotTrainingAccuracy(batches, loss_batch,train_acc_batch, valid_acc_batch):

    loss_plot = plt.subplot(211)
    loss_plot.set_title('Loss')
    loss_plot.plot(batches, loss_batch, 'g')
    loss_plot.set_xlim([batches[0], batches[-1]])
    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')
    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
    acc_plot.set_ylim([0, 1.0])
    acc_plot.set_xlim([batches[0], batches[-1]])
    acc_plot.legend(loc=4)
    plt.tight_layout()
    plt.show()

def run_batch(session, batch_count, network,features, labels):

    accuracy = 0
    for i in range(batch_count):
        batch_start = i * batch_size
        accuracy += session.run(
            network,
            feed_dict={
                input_ph: features[batch_start:batch_start + batch_size],
                labels_ph: labels[batch_start:batch_start + batch_size]
            }
        )

    return accuracy / batch_count

def runNeuralNet(input_ph,labels_ph,oh_train_labels,oh_valid_labels,oh_test_labels,loss,accuracy):

    # Use Adam Optimizer as suggested in lectures
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []
    validation_accuracy = 0.0

    init = tf.initialize_all_variables()
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    session.run(init)

    batch_size = 5 #DEL
    batch_count = int(len(train_features) / batch_size)
    batch_count = 5 #DEL

    for epoch in range(training_epochs):

        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch + 1, training_epochs),
                            unit='batches')

        # The training cycle
        for batch_i in batches_pbar:

            batch_start = batch_i * batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = oh_train_labels[batch_start:batch_start + batch_size]

            _, l = session.run(
                [optimizer, loss],
                feed_dict={input_ph: batch_features, labels_ph: batch_labels})

            if not batch_i % log_batch_step:

                ## Check training & validation accuracy for every specified window
                training_accuracy = session.run( accuracy, feed_dict={input_ph: batch_features, labels_ph: batch_labels}
                )

                idx = np.random.randint(len(valid_features), size=int(batch_size * .2))
                validation_accuracy = session.run(
                    accuracy,
                    feed_dict={input_ph: valid_features[idx, :], labels_ph: oh_valid_labels[idx, :]}
                )

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)


    ## Run against validaton set
    validation_accuracy = run_batch(session, batch_count, accuracy,valid_features, oh_valid_labels)

    ## Run against test data set
    test_accuracy = run_batch(session, batch_count, accuracy,test_features, oh_test_labels)

    print(' Validation accuracy: ', validation_accuracy)
    print(' Test accuracy: ', test_accuracy)

    if plot :
        plotTrainingAccuracy(batches, loss_batch,train_acc_batch, valid_acc_batch)

    ## return session - needed for manual validation...

    return session



def saveModel(sess) :

    print ("Inside save model")
    saver = tf.train.Saver()
    MODEL_SAVE_PATH = "./model/model.ckpt"
    save_path = saver.save(sess, MODEL_SAVE_PATH)
    print("Trained model saved at:", save_path)

def getSession() :

    print ("Inside load model")
    return None
    tf.reset_default_graph()
    saver = tf.train.Saver()
    save_file = './model/model.ckpt'

    import os
    if os.path.isfile(save_file) :
        with tf.Session() as sess:
            saver.restore(sess, save_file)
            # saver.restore(sess, tf.train.latest_checkpoint('.'))
        return sess
    else :
        return None

if __name__ == '__main__':

    print ("Inside main ...")

    #load data
    (train_features, train_labels, test_features, test_labels)  = load()

    #summary of loaded input data set
    n_classes = dataSum(train_features, test_features)

    #explore data ser
    (inputs_per_class, max_inputs) = dataExploration(train_features, train_labels, test_features, test_labels)

    #pre process the data, normalize the data set by creating new data set by slightly modifying available ones
    (train_features, valid_features, train_labels, valid_labels) = \
        preProcess(inputs_per_class, max_inputs,train_features,train_labels)

    #create fully convolutional neural net
    (input_ph, labels_ph, oh_train_labels, oh_valid_labels, loss, accuracy,prediction) \
        = createNeuralNet(train_features, valid_features, train_labels, valid_labels)

    #create test labels
    test_features = np.array(test_features) / 255 * 0.8 + 0.1
    oh_test_labels = tf.one_hot(test_labels, n_classes).eval(session=tf.Session())

    #run the neural net

    session = getSession()

    if session == None :
        session = runNeuralNet(input_ph, labels_ph, oh_train_labels, oh_valid_labels, oh_test_labels, loss, accuracy)


    #predict
    y_pred_cls = tf.argmax(prediction, dimension=1)
    test_cls = np.argmax(oh_test_labels, axis=1)

    #test the accuracy
    print_test_accuracy(session,test_features,oh_test_labels,test_cls,y_pred_cls,
                        show_example_errors=False, show_confusion_matrix=True)

    imgs = ['20.png', '80.png', 'exclamation.png', 'hochwasser.png', 'priority.png']

    new_input = []
    actual_class = [1,2,3,4,5]

    for imgname in imgs:
        image = mpimg.imread('images/' + imgname)
        new_input.append(image)

        if plot :
            plt.imshow(image)
            plt.show()


    new_predictions = session.run(prediction, feed_dict={input_ph: new_input})
    print(new_predictions)

    top_k_probabilities = (session.run(tf.nn.top_k(prediction, k=5), feed_dict={input_ph: new_input}))

    import pandas as pd
    values = np.array([top_k_probabilities])
    indices = np.array([top_k_probabilities])

    for i in range(len(values)) :

        pred_class = indices[i][np.argmax[values[i]]]

        correct_class = np.argmax(actual_class[i])
        plot_title = "Predicted : {} \n Correct : {} ".format(pred_class, correct_class)

        plot_probabilities(indices[i], values[i], plot_title)

        print ("raw top_k results:")
        print ("tf.nn.top_k.values", list(values[i]))
        print ("tf.nn.top_k.indices", list(indices[i]))



