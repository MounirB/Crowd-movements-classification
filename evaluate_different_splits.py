from evaluate_model import *
from confusion_matrix import *
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

from sklearn.metrics import confusion_matrix


def information_extractor(trained_model):
    """
    From the trained_model name, returns the name of the architecture (model type), the split number, and
    the training condition
    """
    model_type = re.findall(".*_?.+3D", trained_model)[0]
    split_number = re.findall("([0-9])_weights\.hdf5", trained_model)[0]
    training_condition = re.findall("_([A-Z]+)[0-9]_", trained_model)[0]

    return model_type, split_number, training_condition


def main_evaluator():
    trained_models_path = "Trained_models/Back_up/5-fold-cross-validation/"
    trained_models_list = os.listdir(trained_models_path)
    batch_size = 1
    information_array = list()

    for trained_model in trained_models_list:
        model_type, split_number, training_condition = information_extractor(trained_model)
        trained_model_path = os.path.join(trained_models_path, trained_model)
        trained_model_data_folder = "Data/Splits/Split"+str(split_number)+"/"
        loss, acc = evaluate(model_type, trained_model_path, trained_model_data_folder, batch_size)
        information_row = [trained_model_path, model_type, training_condition, split_number, loss, acc]
        information_array.append(information_row)

    evaluation_results = pd.DataFrame(information_array, columns=["path", "type", "training_condition", "split_number", "loss", "acc"])
    evaluation_results.to_csv("evaluation_results.csv")

def main_predictor():
    trained_models_path = "Trained_models/Back_up/5-fold-cross-validation/"
    trained_models_list = os.listdir(trained_models_path)
    batch_size = 1
    information_array = list()

    for trained_model in trained_models_list:
        model_type, split_number, training_condition = information_extractor(trained_model)
        trained_model_path = os.path.join(trained_models_path, trained_model)
        trained_model_data_folder = "Data/Splits/Split" + str(split_number) + "/"
        print(trained_model_path, model_type)
        predictions = make_predictions(model_type, trained_model_path, trained_model_data_folder, batch_size)

        information_row = [trained_model_path, model_type, training_condition, split_number, predictions]
        information_array.append(information_row)

    predictions_results = pd.DataFrame(information_array,
                                      columns=["path", "type", "training_condition", "split_number", "predictions"])
    predictions_results.to_csv("predictions_results.csv")

def plot_evaluation_results():
    models_types = ['I3D', 'C3D', 'TWOSTREAM_I3D']
    training_conditions = ['SCRATCH', 'PRETRAINED']
    evaluation_results = pd.read_csv("evaluation_results.csv")
    # fig, axs = plt.subplots(2, 3)
    fig = plt.figure()
    type_counter = 0
    training_counter = 0
    models_accuracies = list()
    for model_type in models_types:
        # models_accuracies = list()
        for training_condition in training_conditions:
            target_model = evaluation_results.loc[(evaluation_results['type'] == model_type) &
                                                      (evaluation_results['training_condition'] == training_condition)]
            model_accuracies = target_model['acc'].values
            print(model_type, training_condition, np.mean(model_accuracies), np.min(model_accuracies), np.max(model_accuracies))
            models_accuracies.append(model_accuracies)
            # axs[training_counter, type_counter].boxplot(model_accuracies)
            # axs[training_counter, type_counter].set_title(model_type+" "+training_condition)
            # axs[training_counter, type_counter].set_ylim([0.2, 0.8])
            training_counter = training_counter + 1

        type_counter = type_counter + 1
        training_counter = 0

    plt.boxplot(models_accuracies)
    plt.ylim([0.2, 0.8])
    plt.ylabel('Accuracy')
    xtick_labels = ['I3D Scratch', 'I3D Pretrained', 'C3D Scratch', 'C3D Pretrained', 'TwoStream-I3D Scratch', 'TwoStream-I3D Pretrained']
    plt.gca().xaxis.set_ticklabels(xtick_labels, rotation=45)
    # fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
    #                     hspace=0.4, wspace=0.5)

    plt.savefig('./BoxPlots_5foldcrossvalidation_crowd11.pdf', bbox_inches='tight')
    plt.show()

def plot_confusion_matrix_results(normalize=False):
    models_types = ['I3D', 'C3D', 'TWOSTREAM_I3D']
    training_conditions = ['SCRATCH', 'PRETRAINED']
    prediction_results = pd.read_csv("predictions_results.csv")
    nb_classes = 11

    type_counter = 0
    training_counter = 0
    for model_type in models_types:
        models_predictions = list()
        for training_condition in training_conditions:
            target_model = prediction_results.loc[(prediction_results['type'] == model_type) &
                                                      (prediction_results['training_condition'] == training_condition)]
            model_confusion_matrix = np.zeros((nb_classes, nb_classes))
            for split_number in target_model['split_number'].values:
                trained_model_data_folder = "Data/Splits/Split" + str(split_number) + "/"
                target_model_split = target_model.loc[target_model['split_number'] == split_number]

                model_predictions = target_model_split['predictions'].values[0]
                model_predictions = [int(label) for label in re.findall("[0-9]+", model_predictions)]
                confusion_matrix = compute_confusion_matrix(trained_model_data_folder, np.array(model_predictions))
                model_confusion_matrix = model_confusion_matrix + confusion_matrix
            title = model_type + " " + training_condition
            if normalize == True:
                labels_distribution = model_confusion_matrix.sum(axis=1)
                model_confusion_matrix = model_confusion_matrix / labels_distribution[:, np.newaxis]
            plot_confusion_matrix(model_confusion_matrix, nb_classes, title, normalize=normalize)
            training_counter = training_counter + 1

        type_counter = type_counter + 1
        training_counter = 0

if __name__ == '__main__':
    main_evaluator()
    plot_evaluation_results()
    main_predictor()
    plot_confusion_matrix_results(normalize=True)
