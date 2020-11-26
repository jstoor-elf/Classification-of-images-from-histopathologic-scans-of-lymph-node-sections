import tensorflow as tf
# Fix gpu configuration if computer has a GPU, mine has not, I set the bool in
# the following conditional to False on my computer
if True:
    tf.config.gpu.set_per_process_memory_fraction(0.3)
    tf.config.gpu.set_per_process_memory_growth(True)

from tensorflow.keras.models import load_model
from tensorflow import keras
from sklearn.metrics import roc_curve, roc_auc_score

import os, sys
import numpy as np

import Utils.utilities as ul
from Utils.dataloader import load_data_as_dict
from Utils.metrics import *
from Utils.plotter import plot_roc_curve


###  MAIN COMMAND TO EVALUATE THE MODEL ###
# python predict.py prediction.json

ABS_PATH = './'


def evaluate_and_predict_model(args, data, custom_objects):

    ''' Make ensamble evaluations and conditionally predictions from models,
        args is a dictionary that can specify if roc curves and the auc score
        shall be calucalted.
    '''

    if args["ensemble"]:
        ensemble_predictions = []
    if args['ROC_AUC']:
        roc_curves = []

    # Evaluate the models indivdually on dataset
    for i, model_name in enumerate(args["modelfiles"]):

        model = load_model(os.path.join(args["model_dir"], model_name), custom_objects)

        # Get metrics scores for specific model
        metrics_names = [m_name if not m_name == 'loss' else model.loss \
            for m_name in model.metrics_names]

        metric_scores = model.evaluate(data[args["dataset_type"]]['images'], \
            data[args["dataset_type"]]['labels'], verbose=0)

        if args["ensemble"] or args['ROC_AUC']:
            predictions = model.predict(data[args["dataset_type"]]['images'])

            if args['ensemble']:
                ensemble_predictions.append(predictions)

            if args['ROC_AUC']:
                fpr, tpr, _ = roc_curve(data[args["dataset_type"]]['labels'], predictions)
                roc_curves.append((fpr, tpr, model.name))
                ns_auc = roc_auc_score(data[args["dataset_type"]]['labels'], predictions)
                metrics_names.append('AUC')
                metric_scores.append(ns_auc)

        # Print metrics for specific model
        print(string_metrics(metric_scores, metrics_names, model.name))

        # Delete model
        del model

    if args["ensemble"] and args['ROC_AUC']:
        return ensemble_predictions, metrics_names, roc_curves
    elif args["ensemble"]:
        return ensemble_predictions, metrics_names, None
    elif args['ROC_AUC']:
        return None, None, roc_curves


def ensamble_evaluation(ensemble_predictions, ground_truth, metrics_names, custom_objects):

    ''' Evaluate the ensembler using all the metrics in metrics_names '''

    # Store metric values
    metric_values = []

    # Use mean for ensemble learner
    mean_pred = (np.sum(ensemble_predictions, axis=0) / len(ensemble_predictions)).squeeze()

    # Convert numpy arrays to tensors
    ground_truth = tf.convert_to_tensor(ground_truth.astype("float32"))
    mean_pred = tf.convert_to_tensor(mean_pred)


    for i, metric_name in enumerate(metrics_names):
        # First check if the function is a custom function
        if metric_name in custom_objects:
            my_metric = custom_objects[metric_name]
        else:
            # Fetch keras metric if it's implemented by the library
            my_metric = keras.metrics.get(metric_name)

        # This was the thing that worked to calculate the accuracy correctly
        if metric_name == 'accuracy':
            score = keras.metrics.binary_accuracy(ground_truth, mean_pred).numpy()
        elif metric_name == 'AUC':
            fpr, tpr, _ = roc_curve(ground_truth.numpy(), mean_pred.numpy())
            score = roc_auc_score(ground_truth.numpy(), mean_pred.numpy())
        else:
            score = my_metric(ground_truth, mean_pred).numpy()

        metric_values.append(score)


    if args['ROC_AUC']:
        return metric_values, (fpr, tpr, 'Ensemble learner')
    else:
        return metric_values, None


def string_metrics(metric_scores, metrics_names, model_name):

    ''' Print the metric scores for the model in question '''

    str = "Final performance on the dataset using {}: \n".format(model_name)
    for i, (metric_name, metric_score) in enumerate(zip(metrics_names, metric_scores)):
        if not metric_name == 'NLL_loss':
            str += "\t {}: {}".format(metric_name, metric_score)
            str +="\n" if not i == len(metric_scores) - 1 else ""
    return str



if __name__ == '__main__':

    # Get arguments from the function call
    args = ul.test_input(sys.argv)

    # Read in the custom functions
    custom_objects = dict(zip(args["my_metrics"], ul.get_metric(args["my_metrics"])))


    # Load data. YOU HAVE TO NORMALIZE THE DATA!!!
    data = load_data_as_dict([args["dataset_type"]])
    data[args["dataset_type"]]['images'] *= 1. / 255


    # Evaluate (and print result) and conditionally predict model, the latter for ensemble learning
    if args["ensemble"] and len(args["modelfiles"]) > 1:

        # Create predictions from an ensemble learner and evaluate these scores
        ensemble_predictions, metrics_names, rc = evaluate_and_predict_model(args, \
                data, custom_objects)

        # Get metrics scores for ensemble leraner
        metric_scores, roc_tuple = ensamble_evaluation(ensemble_predictions, \
                data[args["dataset_type"]]["labels"], metrics_names, custom_objects)

        # Print metrics for ensemble learner
        print(string_metrics(metric_scores, metrics_names, "Ensemble Learner"))

        # Ensemble learner + ROC
        if args['ROC_AUC']:
            rc.append(roc_tuple)
            plot_roc_curve(rc, os.path.join(ABS_PATH, args["roc_path"]), args["roc_figname"])

    elif args['ROC_AUC']:
        # Roc curves and AUC for all models, but no ensemble learner is used
        _, _, rc = evaluate_and_predict_model(args, data)
        plot_roc_curve(rc, os.path.join(ABS_PATH, args["roc_path"]), args["roc_figname"])

    else:
        # Only print metrics for models specified in the list of model .h5 files
        evaluate_and_predict_model(args, data, custom_objects)
