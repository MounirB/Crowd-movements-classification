# Crowd movements classification

This project is used for the classification of 10 crowd movements illustrated in the Crowd-11 dataset. The 11th class is intended for empty scenes.
Three different architectures are employed for the classification : 
- The C3D architecture. Namely the 3D ConvNets that is presented in the following article : [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/pdf/1412.0767.pdf).
```
@inproceedings{tran2015learning,
  title={Learning spatiotemporal features with 3d convolutional networks},
  author={Tran, Du and Bourdev, Lubomir and Fergus, Rob and Torresani, Lorenzo and Paluri, Manohar},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={4489--4497},
  year={2015}
}
```
The implementation of C3D in Keras was forked from [here](https://github.com/axon-research/c3d-keras).

- The I3D architecture and its extension the TwoStream-I3D. Namely the Inflated 3D architecture that is presented in the following article : [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf)

```
@inproceedings{carreira2017quo,
  title={Quo vadis, action recognition? a new model and the kinetics dataset},
  author={Carreira, Joao and Zisserman, Andrew},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6299--6308},
  year={2017}
}
```

The implementation of I3D in Keras was forked from [here](https://github.com/dlpbc/keras-kinetics-i3d).

# Requirements

Install the latest versions of Tensorflow, Keras, and Opencv. You may also need to install Numpy, Matplotlib, Pandas, and Sklearn.

## Downloading the Crowd-11 dataset

Instructions on how to get the Crowd-11 dataset may be found in the following workshop paper : [Crowd-11: A Dataset for Fine Grained Crowd Behaviour Analysis](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w37/papers/Dupont_Crowd-11_A_Dataset_CVPR_2017_paper.pdf)

```
@inproceedings{dupont2017crowd,
  title={Crowd-11: A dataset for fine grained crowd behaviour analysis},
  author={Dupont, Camille and Tobias, Luis and Luvison, Bertrand},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={9--16},
  year={2017}
}
```

# Usage

After downloading the dataset, extract the optical flow and put both of the flow and the rgb clips inside the subfolders of the `Data/Crowd-11/` folder, like this :

```
-- Data/
    -- Crowd-11/
        -- rgb/
        -- flow/
```

The optical flow is extracted using the following script : `Utils/Opticalflow/multithreaded_OF_extraction.py`.

## K-fold cross-validation
To apply k-fold cross-validation, you should run the following script `k_fold_cross_validation.py` that is suitable to split the dataset into multiple folds. By default, the number of folds K is 5 but you can create as many folds as you want.
In the main program of the `k_fold_cross_validation.py` script, you can find that the program relies on `new_preprocessing.csv` spreadsheet. This spreadsheet is an altered version of `preprocessing.csv` that is obtained from the Crowd-11 authors. We named it `new_preprocessing.csv` after removing the names of the missing clips.

After generating the `Splits/` folder, put it under the `Data/` folder before continuing.

The `k_fold_cross_validation.py` script does not launch the training, you have to do it yourself on each different split.

## Training
To train a model, run the following script `train_model.py` like it is displayed in the following example :

```
python3 train_model.py --model_type 'C3D' --training_condition '_SCRATCH' --data_folder "Data/Splits/Split$1/" --split_number $1 --model_weights_rgb 'Trained_models/sports1M_weights_tf.h5' --model_folder 'Trained_models/' --batch_size 30 --epochs 40 --workers 2
```

## Testing
To evaluate a model, run the following script `evaluate_model.py`, like it is displayed in the following example :
```
python3 evaluate_model.py --model_type 'C3D' '--data_folder' 'Data/' '--trained_model_path' 'Trained_models/C3D_PRETRAINED' '--batch_size' 1
```

### Evaluation of k-fold cross-validation :

To evaluate the cross-validation and display boxplots and confusion matrices of all the models, run the following script `evaluate_different_splits.py`.
