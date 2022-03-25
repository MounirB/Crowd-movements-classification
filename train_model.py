import os
import sys
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam, SGD
import keras.backend as K
import traceback
import argparse

from Utils.get_video import *
from Architectures.Inflated_3D import *
from Architectures.Convolutional_3D import *


C3D_PRETRAINED_SPORTS1M = 'sports1M_weights_tf.h5'
# I3D_PRETRAINED_FORCROWD11_RGB = 'rgb_imagenet_and_kinetics'
# I3D_PRETRAINED_FORCROWD11_FLOW = 'flow_imagenet_and_kinetics'

# If downloaded (MESOCENTRE snippet)
I3D_PRETRAINED_FORCROWD11_RGB = 'Trained_models/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
I3D_PRETRAINED_FORCROWD11_FLOW = 'Trained_models/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'

def define_input(model_type):
    """
    Defines the input prototype according to the chosen model type to train
    :param model_type: type of the model to evaluate
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
        print("# I3D sample_input creation :")
        FRAMES_PER_VIDEO = 20
        FRAME_HEIGHT = 224
        FRAME_WIDTH = 224
        FRAME_CHANNEL = 0

        sample_input = np.empty(
            [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    return sample_input


def load_model(model_type, training_condition, model_weights_rgb, model_weights_flow, input_shape, nb_classes):
    """
    Prepares the model for training
    :param model_type: Name of the architecture
    :param training_condition: If to load weights (pretrained) or no (scratch)
    :param model_weights_rgb: Name or path to the pretrained rgb model
    :param model_weights_flow: Name or path to the pretrained flow model
    :param input_shape: structure of the input shape
    :param nb_classes: number of classes
    :return: return the defined model
    """

    if model_type == 'C3D':
        print("C3D training")
        if training_condition == '_PRETRAINED':
            model = get_model(input_shape, num_classes=487, backend='tf')

            # Prepare for fine-tuning
            model.load_weights(model_weights_rgb)
            model.pop()
            model.add(Dense(nb_classes, activation='softmax', name='predictions'))
        else : #_SCRATCH
            model = get_model(input_shape,
                              num_classes=nb_classes,
                              backend='tf')
    elif model_type == 'I3D':
        print("I3D training")
        if training_condition == '_PRETRAINED':
            model = Inception_Inflated3d(include_top=False,
                                         weights=I3D_PRETRAINED_FORCROWD11_RGB,
                                         input_tensor=None,
                                         input_shape=input_shape,
                                         dropout_prob=0.0,
                                         endpoint_logit=True,
                                         classes=nb_classes)
        else: #_SCRATCH
            model = Inception_Inflated3d(include_top=False,
                                         weights=None,
                                         input_tensor=None,
                                         input_shape=input_shape,
                                         dropout_prob=0.0,
                                         endpoint_logit=True,
                                         classes=nb_classes)
    else: # TWOSTREAM_I3D
        print("TwoStream I3D training")
        if training_condition == '_PRETRAINED':
            model = TwoStream_Inception_Inflated3d(include_top=False,
                                         weights=[I3D_PRETRAINED_FORCROWD11_RGB, I3D_PRETRAINED_FORCROWD11_FLOW],
                                         input_tensor=None,
                                         flow_input_shape=input_shape,
                                         rgb_input_shape=input_shape,
                                         dropout_prob=0.0,
                                         endpoint_logit=True,
                                         classes=nb_classes)
        else: #_SCRATCH
            model = TwoStream_Inception_Inflated3d(include_top=False,
                                         weights=None,
                                         input_tensor=None,
                                         flow_input_shape=input_shape,
                                         rgb_input_shape=input_shape,
                                         dropout_prob=0.0,
                                         endpoint_logit=True,
                                         classes=nb_classes)
    return model

def scheduler(epoch, learningrate):
    """
    Scheduler used to reduce the learning rate each 4 epochs
    :param epoch: current epoch index
    :param lr: current learning rate
    :return: new learning rate
    """
    if epoch % 4 == 0 and epoch != 0:
        learningrate = learningrate/10
    return learningrate


def train(model_type, training_condition, split_number, models_folder, model_weights_rgb, model_weights_flow, data_folder, batch_size, Epochs, workers):
    """
    Trains a model from scratch or fine-tunes it
    :param model_type: Name of the architecture to train
    :param training_condition: Type of the training : from scratch or fine-tuning
    :param models_folder: Models folder, needed to be known if we fine-tune a model
    :param model_weights_rgb : Path or name of the rgb model weights
    :param model_weights_flow : Path or name of the flow model weights
    :param data_folder: Folder containing the folder
    :param batch_size: batch size
    :param epochs: the number of epochs
    :return: trained model
    """
    sample_input = define_input(model_type)

    # Read Dataset
    train_data = pd.read_csv(os.path.join(str(data_folder+'train.csv')))
    validation_data = pd.read_csv(os.path.join(str(data_folder+'val.csv')))
    # Split data into random training and validation sets
    # print(train_data)
    nb_classes = len(set(train_data['class']))

    video_train_generator = generate_video_clips(train_data,
                                                 model_type,
                                                 sample_input.shape,
                                                 nb_classes,
                                                 batch_size)
    video_val_generator = generate_video_clips(validation_data,
                                               model_type,
                                               sample_input.shape,
                                               nb_classes,
                                               batch_size)

    # Get Model
    model = load_model(model_type, training_condition, model_weights_rgb, model_weights_flow, sample_input.shape, nb_classes)

    # Callbacks
    checkpoint = ModelCheckpoint(str(models_folder+model_type+training_condition+split_number+'_weights.hdf5'),
                                 monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min',
                                 save_weights_only=True)
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=100)

    if model_type == 'I3D' or model_type == 'TWOSTREAM_I3D':
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=0,
                                           verbose=1)
        callbacks_list = [checkpoint, reduceLROnPlat, earlyStop]
    else:
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=20,
                                           verbose=1, mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-6)
        change_lr = LearningRateScheduler(scheduler, verbose=1)
        callbacks_list = [change_lr, checkpoint, reduceLROnPlat, earlyStop]


    # compile model
    if model_type == 'I3D' or model_type == 'TWOSTREAM_I3D':
        optim = SGD(lr=0.003, momentum=0.9)
    elif model_type == 'C3D':
        optim = SGD(lr=0.003)
    else:
        optim = Adam(lr=1e-4, decay=1e-6)
        optim = SGD(lr=0.003, momentum=0.9, nesterov=True, decay=1e-6)

    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    if os.path.exists(str(models_folder+model_type+training_condition+split_number+'_weights.hdf5')):
        print('Pre-existing model weights found, loading weights.......')
        model.load_weights(str(models_folder+model_type+training_condition+split_number+'_weights.hdf5'))
        print('Weights loaded')

    # model description
    model.summary()

    # train model
    print('Training started....')

    train_steps = len(train_data) // batch_size
    val_steps = len(validation_data) // batch_size

    if workers > 1:
        use_multiprocessing = True
    else:
        use_multiprocessing = False

    history = model.fit_generator(
        video_train_generator,
        steps_per_epoch=train_steps,
        epochs=Epochs,
        validation_data=video_val_generator,
        validation_steps=val_steps,
        verbose=1,
        callbacks=callbacks_list,
        workers=workers,
        use_multiprocessing=use_multiprocessing
    )


def main(args):
    try:
        train(args.model_type,
              args.training_condition,
              args.split_number,
              args.model_folder,
              args.model_weights_rgb,
              args.model_weights_flow,
              args.data_folder,
              args.batch_size,
              args.epochs,
              args.workers)
    except Exception as err:
        print('Error:', err)
        traceback.print_tb(err.__traceback__)
    finally:
        # Destroying the current TF graph to avoid clutter from old models / layers
        K.clear_session()


if __name__ == '__main__':
    ## ensure that the script is running on gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    ## clear session, in case it's necessary
    K.clear_session()

    ## verify that we are running on gpu
    if len(K.tensorflow_backend._get_available_gpus()) == 0:
        print('error-no-gpu')
        exit()
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        help='Specify the model name.',
                        type=str,
                        choices=['TWOSTREAM_I3D', 'C3D', 'I3D'],
                        required=True)

    parser.add_argument('--training_condition',
                        help='Specify how was the weights'' state of the model before going to be trained on Crowd-11.',
                        type=str,
                        choices=['_SCRATCH', '_PRETRAINED'],
                        required=True)

    parser.add_argument('--data_folder',
                        help='Specify the path to the data folder.',
                        type=str,
                        default='Data/',
                        required=True)

    parser.add_argument('--model_folder',
                        help='Specify the path to the trained models.',
                        type=str,
                        default='Trained_models/',
                        required=True)

    parser.add_argument('--model_weights_rgb',
                        help='Specify the name or the path to the RGB weights.',
                        type=str)

    parser.add_argument('--model_weights_flow',
                        help='Specify the name or the path to the FLOW weights.',
                        type=str)

    parser.add_argument('--batch_size',
                        help='Specify the batch_size for training.',
                        type=int,
                        required=True)

    parser.add_argument('--epochs',
                        help='Specify the number of epochs for training.',
                        type=int,
                        required=True)

    parser.add_argument('--split_number',
                        help='Specify the number of the split.',
                        type=str,
                        required=True)

    parser.add_argument('--workers',
                        help='Specify the number of GPUs involved in the computation.',
                        type=int,
                        required=True)
    args = parser.parse_args()

    main(args)
