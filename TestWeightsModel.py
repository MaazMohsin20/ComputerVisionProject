from utils import  LoadDataSet
from keras.optimizers import Adam
from models import resnet_v1
import keras
import  os
from utils import lr_schedule,load_model_display_matrices
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import itertools

wightPath='C:/Users/dell/Downloads/phd/Computer vision cs 867/Project/saved models/saved_models_only_resnet_97.4_acc/Radiography_resnet20_model.020.h5'
batch_size = 32
epochs = 20
data_augmentation = False
num_classes = 3
subtract_pixel_mean = True  # Subtracting pixel mean improves accuracy
base_model = 'resnet20'
# Choose what attention_module to use: cbam_block / se_block / None
attention_module = None
model_type = base_model if attention_module==None else base_model+'_'+attention_module

# Load the CIFAR10 data.
images,labels=LoadDataSet();
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.20)
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

depth = 20 # For ResNet, specify the depth (e.g. ResNet50: depth=50)
model = resnet_v1.resnet_v1(input_shape=input_shape, depth=depth, attention_module=attention_module)

# model = resnet_v2.resnet_v2(input_shape=input_shape, depth=depth, attention_module=attention_module)
# model = resnext.ResNext(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = mobilenets.MobileNet(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = inception_v3.InceptionV3(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = densenet.DenseNet(input_shape=input_shape, classes=num_classes, attention_module=attention_module)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
#plot_model(model, to_file='CBAM.png')
print(model_type)

model.load_weights(filepath=wightPath)

    #test_input = np.asarray(np.vstack((test_pos_fd, test_neg_fd)))
    #test_output = np.asarray(np.hstack((test_pos_label, test_neg_label)))

#print(np.shape(x_test))
y_pred = np.round(model.predict(x_test),0)
#print(confusion_matrix(y_pred.argmax(axis=1), predictions.argmax(axis=1)))
y_pred=y_pred.argmax(axis=1);
y_test=y_test.argmax(axis=1);
#print(y_pred,y_test)
print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
print('\n')
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test, y_pred))
from utils import plot_confusion_matrix
from matplotlib import pyplot as plt
class_names = ['COVID', 'NORMAL', 'Viral_Pneumonia']

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
plt.show()
plt.savefig('Confusion Matrix ResNet')