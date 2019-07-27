import argparse
import pandas as pd
import keras.backend as K
import traceback
from keras.optimizers import Adam, SGD
import os
from Utils.get_video import *
from Architectures.Inflated_3D import Inception_Inflated3d, TwoStream_Inception_Inflated3d
from Architectures.Convolutional_3D import get_model
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def define_input(model_type):
    """
    Defines the input prototype according to the chosen model to evaluate
    :param model_type: name of the model to evaluate
    :return: returns the input prototype
    """

    if model_type == 'C3D':
        print("# C3D sample_input creation :")
        FRAMES_PER_VIDEO = 16
        FRAME_HEIGHT = 112
        FRAME_WIDTH = 112
        FRAME_CHANNEL = 3

        sample_input = np.empty(
            [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)
    elif model_type == 'I3D':
        print("# I3D sample_input creation :")
        FRAMES_PER_VIDEO = 20
        FRAME_HEIGHT = 224
        FRAME_WIDTH = 224
        FRAME_CHANNEL = 3

        sample_input = np.empty(
            [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    else: # TWOSTREAM_I3D
        print("# TWOSTREAM_I3D sample_input creation :")
        FRAMES_PER_VIDEO = 20
        FRAME_HEIGHT = 224
        FRAME_WIDTH = 224
        FRAME_CHANNEL = 0

        sample_input = np.empty(
            [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    return sample_input

def load_model(model_type, model_weights_path, input_shape, nb_classes):
    """
    Loads a model from the models folder
    :param model_type: model type
    :param model_weights_path: path to the trained model
    :param input_shape: prototype of the input data
    :param nb_classes: the number of classes of the dataset
    :return: returns the loaded model
    """

    if model_type == 'C3D':
        print("C3D evaluation")
        model = get_model(input_shape,
                          num_classes=nb_classes,
                          backend='tf')

        model.load_weights(model_weights_path)
    elif model_type == 'I3D':
        print("I3D evaluation")
        model = Inception_Inflated3d(include_top=False,
                                     weights=None,
                                     input_tensor=None,
                                     input_shape=input_shape,
                                     dropout_prob=0.0,
                                     endpoint_logit=True,
                                     classes=nb_classes)
        model.load_weights(model_weights_path)
    else: # TWOSTREAM_I3D
        print("TWOSTREAM_I3D evaluation")
        model = TwoStream_Inception_Inflated3d(include_top=False,
                                         weights=None,
                                         input_tensor=None,
                                         flow_input_shape=input_shape,
                                         rgb_input_shape=input_shape,
                                         dropout_prob=0.0,
                                         endpoint_logit=True,
                                         classes=nb_classes)
        model.load_weights(model_weights_path)

    return model

def compute_plot_confusion_matrix(data_folder, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Load test data
    test_data = pd.read_csv(os.path.join(str(data_folder + 'test.csv')))

    # Determine the number of classes
    nb_classes = len(set(test_data['class']))

    # Determine class labels
    classes = np.arange(0, nb_classes)

    # Target labels
    y_true = test_data['class'].values

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

def compute_confusion_matrix(data_folder, y_pred,
                          normalize=False):
    """
    This function returns the computed confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Load test data
    test_data = pd.read_csv(os.path.join(str(data_folder + 'test.csv')))

    # Determine the number of classes
    nb_classes = len(set(test_data['class']))

    # Determine class labels
    classes = np.arange(0, nb_classes)

    # Target labels
    y_true = test_data['class'].values

    # Compute confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)

    if normalize:
        confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
    #     # print('Confusion matrix, without normalization')

    return confusion_mat

def plot_confusion_matrix(confusion_matrix, nb_classes, title, normalize=False, cmap=plt.cm.Blues):
    classes = np.arange(0, nb_classes)

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else '.0f'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('./'+title+'_confusionmatrix.pdf', bbox_inches='tight')
    plt.show()


def make_predictions(model_type, trained_model_path, data_folder, batch_size):
    """
    Makes predictions of video clips from the testing set of the dataset
    :param model_type: mentions the name of the architecture from which the model was trained
    :param trained_model_path: path to a trained model of the architecture
    :param data_folder: path to the dataset folder
    :param batch_size: batch size
    :return: returns the predictions list for all the testing set
    """
    # Load test data
    test_data = pd.read_csv(os.path.join(str(data_folder + 'test.csv')))

    # Determine the number of classes
    nb_classes = len(set(test_data['class']))
    # Define input
    sample_input = define_input(model_type)

    # Load trained model
    model = load_model(model_type, trained_model_path, sample_input.shape, nb_classes)
    model.summary()
    model.compile(optimizer=SGD(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])

    # Predict labels from video clips
    x_flow_clips_paths = test_data['x_axis_flowclips_path'].values
    y_flow_clips_paths = test_data['y_axis_flowclips_path'].values
    rgb_clips_paths = test_data['rgbclips_path'].values

    predictions = list()
    if model_type == 'TWOSTREAM_I3D':
        for x_flow_clip_path, y_flow_clip_path, rgb_clip_path in zip(x_flow_clips_paths, y_flow_clips_paths, rgb_clips_paths):
            video_clip = generate_video_sample([rgb_clip_path, x_flow_clip_path, y_flow_clip_path], sample_input.shape)
            label_probabilities = model.predict(video_clip)
            predictions.append(np.argmax(label_probabilities))
            print(np.argmax(label_probabilities))

    else: #C3D, I3D
        for rgb_clip_path in rgb_clips_paths:
            video_clip = generate_video_sample(rgb_clip_path, sample_input.shape)
            label_probabilities = model.predict(video_clip)
            predictions.append(np.argmax(label_probabilities))
            print(np.argmax(label_probabilities))

    return predictions


def main(args):
    try:
        predictions = make_predictions(args.model_type,
                 args.trained_model_path,
                 args.data_folder,
                 args.batch_size)
        # Compute and plot confusion matrix
        plot_confusion_matrix(args.data_folder, np.array(predictions), normalize=True)
    except Exception as err:
        print('Error:', err)
        traceback.print_tb(err.__traceback__)
    finally:
        # Destroying the current TF graph to avoid clutter from old models / layers
        K.clear_session()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        help='Specify the model name.',
                        type=str,
                        choices=['TWOSTREAM_I3D', 'C3D', 'I3D'],
                        default='C3D',
                        required=True)

    parser.add_argument('--data_folder',
                        help='Specify the dataset path.',
                        type=str,
                        default='Data/',
                        required=True)

    parser.add_argument('--trained_model_path',
                        help='Specify the model path.',
                        type=str,
                        default='Trained_models/',
                        required=True)

    parser.add_argument('--batch_size',
                        help='Specify the batch_size for evaluation.',
                        type=int,
                        default=1)
    args = parser.parse_args()
    main(args)
