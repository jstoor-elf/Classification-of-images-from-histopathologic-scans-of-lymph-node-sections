from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import os, sys, json
import pathlib
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..', 'Models')))
import Models as M
from .metrics import *

ABS_PATH = './'


def get_model(input_shape, model_name, specs):

    ''' Load a model from a dictionary

    # Arguments:
        input_shape: input size is the size of the image with color channel
        model_name: The name of the model, can be of type:
            - DenseNet
            - ResNet
            - VGG
        specs: Is a dictionary containing arguments to the specific model
    '''

    if model_name == "DenseNet":
        return M.densenet.dense_net(
            input_shape,
            base=specs["base"],
            dense_blocks=specs["dense_blocks"],
            blocks_per_layer=specs["layers_per_block"],
            growth_rate=specs["growth_rate"],
            classes=specs["classes"],
            out_fnc=specs["out_fnc"],
            dense_depth=specs["dense_depth"],
            bottleneck=specs["bottleneck"],
            dropout_rate=specs["dropout_rate"]
        )

    elif model_name == "ResNet":
        return M.resnet.res_net(
            input_shape,
            Base=specs["base"],
            layers_per_block=specs["layers_per_block"],
            out_fnc=specs["out_fnc"],
            n_classes=specs["n_classes"]
        )

    elif model_name== "VGG":
        return M.vgg.vgg_net(
            input_shape,
            Base=specs["base"],
            basefactors=specs["basefactors"],
            layers=specs["layers"],
            fulllayers=specs["fulllayers"],
            out_c=specs["out_c"],
            out_fnc= specs["out_fnc"],
            batchnorm2D=specs["batchnorm2D"],
            dropout=specs["dropout"],
            maxpool=specs["maxpool"]
        )
    else:
        raise Exception("Model not implemented")


def get_loss_metrics_optim(train_params):

    ''' Get metrics, loss function and optimizer '''

    metrics = get_metric(train_params["metrics"])
    loss = get_loss_function(train_params["loss"])
    optimizer = fetch_optimizer(train_params["optimizer"])
    return metrics, loss, optimizer


# No custom loss function is implemented as of now
def get_loss_function(loss):

    ''' Returns loss function or its string representation if the function is
        implemented in keras
    '''
    return loss



def metrics(metric):

    ''' Returns the evaluation metric or its string representation if the
        function is implemented in keras
    '''

    if metric == "NLL_loss":
        return NLL_loss
    elif metric == 'sensitivity':
        return sensitivity
    elif metric == 'sepcificity':
        return sepcificity
    elif metric == 'precision':
        return precision
    elif metric == 'negative_prediction':
        return negative_prediction
    elif metric == 'miss_rate':
        return miss_rate
    elif metric == 'fall_out':
        return fall_out
    elif metric == 'false_discovery':
        return false_discovery
    elif metric == 'false_omission':
        return false_omission
    return metric


def get_metric(my_metrics):

    ''' Returns loss metrics as a list

    # Arguments:
        my_metrics: List of string representations of the metrics
    '''

    M = []
    for metric in my_metrics:
        M.append(metrics(metric))
    return M


def test_input(arra):

    ''' Test system input, shoud be json file. Return it as a dictionary '''

    if  len(arra) > 0 and len(arra) < 2:
        raise Exception("Wrong number of input")

    # Check correctness of input file
    filename = os.path.join('InputFiles/', arra[1])
    if not os.path.isfile(filename):
        raise Exception("Input file doesn't exist!")

    # Read in the file
    with open(filename) as json_file:
        args = json.load(json_file)

    return args


def fetch_optimizer(name):

    ''' Fetch the optimizer

    # Arguments:
        name: Can be of type "Adam", "SGD" or "RMSprop"
    '''

    if name == 'Adam':
        Optim = Adam
    elif name == 'SGD':
        Optim = SGD
    elif name == 'RMSprop':
        Optim = RMSprop
    else:
        raise Exception('The given optimizer is not defined')

    return Optim


def setup_dir(paths):

    ''' Set up directory if it doesn't exist. Works for subdirectories also '''

    for path in paths:
        full_path = os.path.join(ABS_PATH, path)
        if not os.path.exists(full_path):
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)


# Create callbacks used for training
def create_callbacks(specs, history_path="Results/History", model_path="Results/Models"):

    # Set up directories if they don't exist
    setup_dir([history_path, model_path])

    csv_logger = CSVLogger(os.path.join(ABS_PATH, history_path, specs["history_file_name"]), append=False)
    chk = ModelCheckpoint(os.path.join(ABS_PATH, model_path, specs["model_file_name"]),
            monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    return [chk, csv_logger]
