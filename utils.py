import itertools

import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
def resize_Images(list_Images):

    list_resized_img=[]
    for x in list_Images:
        resized_image = cv2.resize(x, (96, 160))
        list_resized_img.append(resized_image)
    return list_resized_img
def pil_load_images_from_folder(folder,img_witdh,img_height):
    images_rgb = []
    images_name = []
    for filename in os.listdir(folder):
        newsize = (img_witdh, img_height)
        img_rgb = Image.open(fp=os.path.join(folder, filename),mode='r')
        img_rgb = img_rgb.resize(newsize)
        images_name.append(filename)
        if img_rgb is not None:
            images_rgb.append(img_rgb)
    return images_rgb,images_name
    # This method will show imag


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
def LoadDataSet():
    class_names = ['COVID', 'NORMAL', 'Viral_Pneumonia']
    class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
    nb_classes = len(class_names)
    """
         Load the data:
             - 14,034 images to train the network.
             - 3,000 images to evaluate how accurately the network learned to classify images.
     """

    datasets = ['C:/Users/dell/Desktop/ds project/CV Project\COVID-19 Radiography Database/dataset']
    #datasets = ['C:/Users/dell/Desktop/ds project/CV Project/dataset']

    output = []
    IMAGE_SIZE = (224, 224)

    # Iterate through training and test sets
    for dataset in datasets:

        images = []
        labels = []

        print("Loading {}".format(dataset))

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return images,labels

def TestSavedModel(model,wightPath):

    #load and use weights from a checkpoint
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import ModelCheckpoint
    import matplotlib.pyplot as plt
    import numpy
    # create model
    # load weights
    model.load_weights(wightPath)
    # Compile model (required to make predictions)
    # load pima indians dataset
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # estimate accuracy on whole dataset using loaded weights
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
def load_model_display_matrices(model,filepath,testx,testy):
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    #test_input = np.asarray(np.vstack((test_pos_fd, test_neg_fd)))
    #test_output = np.asarray(np.hstack((test_pos_label, test_neg_label)))
    y_pred = model.predict(testx)
    model.load_weights(filepath)
    print("Accuracy: " + str(accuracy_score(testy, y_pred)))
    print('\n')
    print(classification_report(testy, y_pred))