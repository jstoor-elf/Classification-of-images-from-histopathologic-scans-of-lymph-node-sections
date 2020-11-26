import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from .utilities import setup_dir


def get_keys(history):

    ''' Get the keys to be used in the plotting function '''

    metric_names = []
    # Histories is a dict
    for key in history:
        if not 'val_' in key:
            metric_names.append(key)
    return metric_names


def check_keys(histories):

    ''' Check if all dicts in histories have the same keys '''

    for base_hist in histories[0]:
        for history in histories[:1]:
            if not base_hist in history:
                return False
    return True


def plot_data(histories, network_names, path, fig_name, metric, train=False):

    ''' Plot the learning curves for a specified metric for different networks '''

    # Check if histories have the same keys
    if not check_keys(histories):
        raise Exception("The csv files haven't used the same metrics!")
    if not metric in get_keys(histories[0]): # Get the metric names
        raise Exception("The metric was not trained on the network!")

    cmap = matplotlib.cm.get_cmap('rainbow')
    colors = np.arange(1. / len(histories), 1.0, 1. / len(histories))

    plt.figure(num=None, figsize=(6, 5))
    for j, history in enumerate(histories):
        if train:
            plt.plot(history[metric], linestyle='--', color=cmap(colors[j]), \
                label="{} train".format(network_names[j]))
            plt.plot(history["val_" + metric], color=cmap(colors[j]), \
                label="{} valid".format(network_names[j]))
        else:
            plt.plot(history["val_" + metric], color=cmap(colors[j]), \
                label="{}".format(network_names[j]))

    plt.xlabel("epochs")
    str = "Training and validation" if train else "Validation"
    plt.ylabel("{} {}".format(str ,metric))
    plt.legend(loc="upper right") if metric == "loss" or metric == "NLL_loss" \
            else plt.legend(loc="lower right")


    setup_dir([path]) # Set up the directory of it doesn't exist
    plt.savefig(os.path.join(path, fig_name), pad_inches=0.1)
    plt.close()



def plot_roc_curve(roc_curves, path, fig_name):

    ''' Plot ROC curve for for different model

    # Arguments:
        roc_curves: a list of tuples. The first arguments of each tuple is a list
            of the false positive rates for different thresholds, the second argument
            is the true positive rates for different thresholds, and the third
            arguemnt is the name of the model.
        path_name: Path to where to store the roc curve
    '''

    plt.figure(num=None, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, label='Chance', alpha=.5)

    for (fpr, tpr, nm) in roc_curves:
        plt.plot(fpr, tpr, label=nm)

    plt.legend(ncol=2, loc="lower right")
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")

    setup_dir([path]) # Set up the directory of it doesn't exist
    plt.savefig(os.path.join(path, fig_name), pad_inches=0.1)
    plt.close()
