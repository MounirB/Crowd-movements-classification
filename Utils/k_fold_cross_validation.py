import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import os
import re
import csv


def verify_folds_intersection(folds_list):
    """
    Used to verify that the folds scenes do not overlap
    """
    for num_fold in range(0, len(folds_list)):
        selected_fold_scenes = folds_list[num_fold]
        for num_concurrent_fold in range(0, len(folds_list)):
            if num_fold == num_concurrent_fold:
                continue
            else:
                print(set(folds_list[num_fold]).intersection(set(folds_list[num_concurrent_fold])))

def sum_folds_lengths(folds_list):
    """
    Verify that we retrieve the number of scenes via the sum of the lengths of the folds
    """
    summation = 0
    for num_fold in range(0, len(folds_list)):
        summation = summation + len(folds_list[num_fold])
    return summation


def each_fold_length(folds_list):
    """
    Display each fold's length
    """
    for num_fold in range(0, len(folds_list)):
        print(len(folds_list[num_fold]))


def make_train_valid_test_splits(database, folds_scenes, nb_folds):
    """
    Make train validation test splits according to the number of folds.
    We do not intend to explore all the possible combinations.
    Test and validation folds are K times randomly chosen.
    Validation fold following the test fold.
    """
    test_numfolds_NotAlreadySelected = list(range(0, nb_folds))
    val_numfolds_NotAlreadySelected = list(range(0, nb_folds))

    temporary_val_numfolds_NotAlreadySelected = list()


    SplitsFolder = "Splits/"

    for iter in range(0, nb_folds):
        SplitFolder = SplitsFolder+"Split"+str(iter)+"/"
        train_num_folds = list(range(0, nb_folds))
        test_numfold = rd.choice(test_numfolds_NotAlreadySelected)

        temporary_val_numfolds_NotAlreadySelected = val_numfolds_NotAlreadySelected.copy()

        if test_numfold in temporary_val_numfolds_NotAlreadySelected:
            temporary_val_numfolds_NotAlreadySelected.remove(test_numfold)

        val_numfold = rd.choice(temporary_val_numfolds_NotAlreadySelected)

        print(test_numfold)
        print(val_numfold)

        train_num_folds.remove(test_numfold)
        train_num_folds.remove(val_numfold)
        test_numfolds_NotAlreadySelected.remove(test_numfold)
        val_numfolds_NotAlreadySelected.remove(val_numfold)

        test_fold = folds_scenes[test_numfold]
        val_fold = folds_scenes[val_numfold]
        train_folds = list()
        for num_fold in train_num_folds:
            train_folds = train_folds + folds_scenes[num_fold]

        # print("test_split:", test_fold)
        # print("val_split:", val_fold)
        # print("train_split:", train_folds)

        # os.mkdir(SplitFolder)
        create_csvs(SplitFolder, database, train_folds, val_fold, test_fold)

def create_csvs(Splitfolder, db, train_scenes, validation_scenes, test_scenes):
    """
    Generate the csv files
    """
    # Initialization

    Utils_folder = 'Utils/'
    dataset_split_directory = "Data/"
    relative_rgb_dataset_directory = "../Data/Crowd-11/rgb/"
    relative_flow_dataset_directory = "../Data/Crowd-11/flow/"
    written_rgb_dataset_directory = "Data/Crowd-11/rgb/"
    written_flow_dataset_directory = "Data/Crowd-11/flow/"

    videos = os.listdir(relative_rgb_dataset_directory)
    flowvideos = os.listdir(relative_flow_dataset_directory)

    # Splitting the dataset
    labels = [re.findall("^[0-9][0-9]?", video)[0] for video in videos]
    # train csv
    train_database = db.loc[db['scene_number'].isin(train_scenes)]
    train_video_names = [re.findall("(.*)\.[ma][pv][4i]", video_name)[0] for video_name in
                         train_database['video_name'].values]
    train_rgbvideos = [video for video in videos if
                    re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)\.[ma][pv][4i]", video)[0] in train_video_names]
    train_xflowvideos = list()
    train_yflowvideos = list()
    for video in flowvideos:
        if re.findall('.*_x\.[ma][pv][i4]', video):
            train_xflowvideos.append(video)
    for video in flowvideos:
        if re.findall('.*_y\.[ma][pv][i4]', video):
            train_yflowvideos.append(video)

    train_rgbvideos = sorted(train_rgbvideos)
    train_yflowvideos = [video for video in train_yflowvideos if
                         re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)_y\.[ma][pv][4i]", video)[0] in train_video_names]
    train_yflowvideos = sorted(train_yflowvideos)
    train_xflowvideos = [video for video in train_xflowvideos if
                         re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)_x\.[ma][pv][4i]", video)[0] in train_video_names]
    train_xflowvideos = sorted(train_xflowvideos)
    train_labels = [re.findall("^[0-9][0-9]?", video)[0] for video in train_rgbvideos]
    train = [[str(written_rgb_dataset_directory + rgbvideo), str(written_flow_dataset_directory + xflowvideo),
              str(written_flow_dataset_directory + yflowvideo), label] for rgbvideo, xflowvideo, yflowvideo, label in
             zip(train_rgbvideos, train_xflowvideos, train_yflowvideos, train_labels)]

    train_labels = [int(label) for label in train_labels]
    plt.hist(train_labels, bins=11)
    plt.show()
    print(train)

    # val csv
    val_database = db.loc[db['scene_number'].isin(validation_scenes)]
    val_video_names = [re.findall("(.*)\.[ma][pv][4i]", video_name)[0] for video_name in
                         val_database['video_name'].values]
    val_rgbvideos = [video for video in videos if
                       re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)\.[ma][pv][4i]", video)[0] in val_video_names]
    val_xflowvideos = list()
    val_yflowvideos = list()
    for video in flowvideos:
        if re.findall('.*_x\.[ma][pv][i4]', video):
            val_xflowvideos.append(video)
    for video in flowvideos:
        if re.findall('.*_y\.[ma][pv][i4]', video):
            val_yflowvideos.append(video)

    val_rgbvideos = sorted(val_rgbvideos)
    val_yflowvideos = [video for video in val_yflowvideos if
                         re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)_y\.[ma][pv][4i]", video)[
                             0] in val_video_names]
    val_yflowvideos = sorted(val_yflowvideos)
    val_xflowvideos = [video for video in val_xflowvideos if
                         re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)_x\.[ma][pv][4i]", video)[
                             0] in val_video_names]
    val_xflowvideos = sorted(val_xflowvideos)
    val_labels = [re.findall("^[0-9][0-9]?", video)[0] for video in val_rgbvideos]
    val = [[str(written_rgb_dataset_directory + rgbvideo), str(written_flow_dataset_directory + xflowvideo),
              str(written_flow_dataset_directory + yflowvideo), label] for rgbvideo, xflowvideo, yflowvideo, label in
             zip(val_rgbvideos, val_xflowvideos, val_yflowvideos, val_labels)]

    val_labels = [int(label) for label in val_labels]
    plt.hist(val_labels, bins=11)
    plt.show()
    print(val)

    # test csv
    test_database = db.loc[db['scene_number'].isin(test_scenes)]
    test_video_names = [re.findall("(.*)\.[ma][pv][4i]", video_name)[0] for video_name in
                       test_database['video_name'].values]
    test_rgbvideos = [video for video in videos if
                     re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)\.[ma][pv][4i]", video)[0] in test_video_names]
    test_xflowvideos = list()
    test_yflowvideos = list()
    for video in flowvideos:
        if re.findall('.*_x\.[ma][pv][i4]', video):
            test_xflowvideos.append(video)
    for video in flowvideos:
        if re.findall('.*_y\.[ma][pv][i4]', video):
            test_yflowvideos.append(video)

    test_rgbvideos = sorted(test_rgbvideos)
    test_yflowvideos = [video for video in test_yflowvideos if
                       re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)_y\.[ma][pv][4i]", video)[
                           0] in test_video_names]
    test_yflowvideos = sorted(test_yflowvideos)
    test_xflowvideos = [video for video in test_xflowvideos if
                       re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*)_x\.[ma][pv][4i]", video)[
                           0] in test_video_names]
    test_xflowvideos = sorted(test_xflowvideos)
    test_labels = [re.findall("^[0-9][0-9]?", video)[0] for video in test_rgbvideos]
    test = [[str(written_rgb_dataset_directory + rgbvideo), str(written_flow_dataset_directory + xflowvideo),
            str(written_flow_dataset_directory + yflowvideo), label] for rgbvideo, xflowvideo, yflowvideo, label in
           zip(test_rgbvideos, test_xflowvideos, test_yflowvideos, test_labels)]

    test_labels = [int(label) for label in test_labels]
    plt.hist(test_labels, bins=11)
    plt.show()
    print(test)

    # create_csvs
    with open(str(Splitfolder + 'train.csv'), 'w', newline='') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',')
        mywriter.writerow(['rgbclips_path', 'x_axis_flowclips_path', 'y_axis_flowclips_path', 'class'])
        for video in train:
            mywriter.writerow(video)
        print('Training CSV file created successfully')

    with open(str(Splitfolder + 'val.csv'), 'w', newline='') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',')
        mywriter.writerow(['rgbclips_path', 'x_axis_flowclips_path', 'y_axis_flowclips_path', 'class'])
        for video in val:
            mywriter.writerow(video)
        print('Validation CSV file created successfully')

    with open(str(Splitfolder + 'test.csv'), 'w', newline='') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',')
        mywriter.writerow(['rgbclips_path', 'x_axis_flowclips_path', 'y_axis_flowclips_path', 'class'])
        for video in test:
            mywriter.writerow(video)
        print('Testing CSV file created successfully')

    print('CSV files created successfully')


def folds_histograms(database, folds_list):
    """
    Displays the histogram of each fold.
    Shows the frequency of each label for each fold.
    """
    dataset_directory = "../Data/Crowd-11/"

    videos = os.listdir(dataset_directory)
    for fold_scenes in folds_list:
        fold_database = database.loc[database['scene_number'].isin(fold_scenes)]
        fold_videos = [video for video in videos if
                       re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*?\.[ma][pv][4i])", video)[0] in fold_database[
                           'video_name'].values]
        fold_labels = [int(re.findall("^[0-9][0-9]?", video)[0]) for video in fold_videos]
        plt.hist(fold_labels, bins=11)
        plt.show()


def scenes_counter(all_scenes_set, all_scenes_list):
    """
    Counting the number of sequences per scene
    """
    all_scenes_count = list()
    for scene in list(all_scenes_set):
        all_scenes_count.append(all_scenes_list.count(scene))
    return all_scenes_count
    # print(sorted(list(set(all_scenes_count))))
    # plt.hist(all_scenes_count, bins=len(list(set(all_scenes_count))))
    # plt.show()

def sort_scenes_according_to_nbsequences(all_scenes_set, all_scenes_list):
    all_scenes_count = scenes_counter(all_scenes_set, all_scenes_list)


def classes_counter(database, nb_classes):
    """
    Provides the sequences frequency for each class
    """
    classes = list(database['label'].values)
    ### Class occurrences counter
    frequences = [classes.count(int(label)) for label in range(0, nb_classes)]
    return frequences


def determine_min_score_fold_index(folds_distribs, nb_classes):
    """
    Returns the index of the fold with the smallest distribution score
    """
    folds_scores = [sum(distrib)/nb_classes for distrib in folds_distribs]
    # print(folds_scores)
    min_score_index = folds_scores.index(min(folds_scores))
    # print(min_score_index)
    return min_score_index

def update_fold_distribution(dataset_directory, fold_distribution, num_scene, dataframe, frequences, nb_folds):
    """
    Updates the folds weights according to the number of labels each of their scenes possess.
    """

    videos = os.listdir(dataset_directory)
    scene_database = dataframe.loc[dataframe['scene_number'] == num_scene]
    scene_videos = [video for video in videos if
                   re.findall("[0-9][0-9]?_[0-9]+_[0-9][0-9]?_(.*?\.[ma][pv][4i])", video)[0] in scene_database[
                       'video_name'].values]
    scene_labels = [int(re.findall("^[0-9][0-9]?", video)[0]) for video in scene_videos]
    for scene_label in scene_labels:
        fold_distribution[scene_label] = fold_distribution[scene_label] + 1/(frequences[scene_label]/nb_folds)

    return fold_distribution


if __name__ == "__main__":
    database = pd.read_csv('new_preprocessing.csv')
    nb_classes = 11
    nb_folds = 5
    dataset_directory = "../Data/Crowd-11/rgb/"

    # Number of existing scenes and list of sets of scenes
    all_scenes_set = list(set(database['scene_number'].values))
    all_scenes_list = list(database['scene_number'].values)
    nb_scenes = len(all_scenes_set)

    scenes_frequencies = scenes_counter(all_scenes_set, all_scenes_list)
    folds_distrib = np.zeros((nb_folds, nb_classes))
    total_classes_frequences = classes_counter(database, nb_classes)
    folds_scenes = list(np.zeros(nb_folds))

    for num_fold in range(0, nb_folds):
        folds_scenes[num_fold] = list()

    while all_scenes_set:
        fold_smallest_index = determine_min_score_fold_index(folds_distrib, nb_classes)
        biggest_scene_index = scenes_frequencies.index(max(scenes_frequencies))
        biggest_scene_number = all_scenes_set[biggest_scene_index]

        all_scenes_set.pop(biggest_scene_index)
        scenes_frequencies.pop(biggest_scene_index)

        folds_scenes[fold_smallest_index].append(biggest_scene_number)
        folds_distrib[fold_smallest_index] = update_fold_distribution(dataset_directory, folds_distrib[fold_smallest_index], biggest_scene_number, database, total_classes_frequences, nb_folds)
        print(len(all_scenes_set))
        # print([len(fold) for fold in folds_scenes])

    # ## Folds intersection verification
    # verify_folds_intersection(folds_scenes)
    #
    # ## Folds histograms
    # folds_histograms(database, folds_scenes)

    make_train_valid_test_splits(database, folds_scenes, nb_folds)
