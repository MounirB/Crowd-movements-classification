import numpy as np
import os
import pandas as pd
import argparse
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD
import keras.backend as K
import traceback

from Utils.get_video import *

from Architectures.Inflated_3D import Inception_Inflated3d, TwoStream_Inception_Inflated3d
from Architectures.Convolutional_3D import get_model


def define_input(model_type):
    """
    Defines the input prototype according to the chosen model to evaluate
    :param model_type: architecture's type of the model to evaluate
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
    :param model_type: model type to load
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

def evaluate(model_type, trained_model_path, data_folder, batch_size):
    """
    Evaluates a trained model on the test set of Crowd-11
    :param model_type: the model type to train, it must be stored on a .hdf5 file
    :param trained_model_path: mentions the path to the trained model
    :param test_data: load the test set
    :param batch_size: determine the batch size
    :return: prints the accuracy and the loss ['loss', 'acc']
    """
    # Load data
    test_data = pd.read_csv(os.path.join(str(data_folder + 'test.csv')))

    # Determine the number of classes
    nb_classes = len(set(test_data['class']))

    # Define input
    sample_input = define_input(model_type)

    # Load trained model
    model = load_model(model_type, trained_model_path, sample_input.shape, nb_classes)
    model.summary()
    # model.compile(optimizer=Adam(lr=1e-4, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=SGD(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])

    # Evaluate the model
    if model_type == 'C3D':
        print("C3D model creation :")
        video_test_generator = generate_video_clips(test_data,
                                         model_type,
                                         sample_input.shape,
                                         nb_classes,
                                         batch_size=batch_size)
        [loss, acc] = model.evaluate_generator(video_test_generator, steps=1000, max_queue_size=10, workers=1,
                                               use_multiprocessing=False, verbose=1)
    elif model_type == 'I3D':
        print("I3D model creation :")

        video_test_generator = generate_video_clips(test_data,
                                         model_type,
                                         sample_input.shape,
                                         nb_classes,
                                         batch_size=batch_size)
        [loss, acc] = model.evaluate_generator(video_test_generator, steps=1000, max_queue_size=10, workers=1,
                                               use_multiprocessing=False, verbose=1)
    else: # TWOSTREAM_I3D
        print("TWOSTREAM_I3D model creation :")

        video_test_generator = generate_video_clips(test_data,
                                         model_type,
                                         sample_input.shape,
                                         nb_classes,
                                         batch_size=batch_size)
        [loss, acc] = model.evaluate_generator(video_test_generator, steps=1000, max_queue_size=10, workers=1,
                                               use_multiprocessing=False, verbose=1)

    # Print the evaluation results
    print(model.metrics_names)
    print(loss, acc)

    return loss, acc

def main(args):
    try:
        evaluate(args.model_type,
                 args.trained_model_path,
                 args.data_folder,
                 args.batch_size)
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
                        help='Specify the architecture type of the model.',
                        type=str,
                        choices=['TWOSTREAM_I3D', 'C3D', 'I3D'],
                        default='C3D',
                        required=True)

    parser.add_argument('--data_folder',
                        help='Specify the data folder where the CSV file can be found.',
                        type=str,
                        default='Data/',
                        required=True)

    parser.add_argument('--trained_model_path',
                        help='Specify the trained model path.',
                        type=str,
                        required=True)

    parser.add_argument('--batch_size',
                        help='Specify the batch_size for evaluation.',
                        type=int,
                        default=1)
    args = parser.parse_args()
    main(args)
