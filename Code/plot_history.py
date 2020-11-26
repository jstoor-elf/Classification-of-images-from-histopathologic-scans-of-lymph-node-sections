from argparse import ArgumentParser
import Utils.utilities as ul
import Utils.dataloader as dl
import Utils.plotter as pl
import os


###  COMMANDS  ###
# python plot_history.py -f VGG16.csv ResNet50.csv DenseNet121.csv -n VGGNet16 ResNet50 DenseNet121 -m loss -s loss.png
# python plot_history.py -f VGG16.csv ResNet50.csv DenseNet121.csv -n VGGNet16 ResNet50 DenseNet121 -m binary_accuracy -s accuracy.png
# python plot_history.py -f VGG16.csv ResNet50.csv DenseNet121.csv -n VGGNet16 ResNet50 DenseNet121 -m precision -s precision.png
# python plot_history.py -f VGG16.csv ResNet50.csv DenseNet121.csv -n VGGNet16 ResNet50 DenseNet121 -m sensitivity -s sensitivity.png



ABS_PATH = './'


def get_args():

    ''' Parser that can be used with any of the implemented network models '''

    parser = ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+', help='Include at least one csv file', required=True)
    parser.add_argument('-n', '--networknames', nargs='+', help='Name networks', required=True)
    parser.add_argument('-m', '--metric', dest='metric', help='Which metric to plot', required=True)
    parser.add_argument('-s', '--imagename', dest='imagename', help='name of figure', required=True)
    parser.add_argument('-d', '--file-dir', default='Results/History/', dest='file_dir', help='Directory of csv file')
    parser.add_argument('-i', '--image-dir', default='Results/Plots/', dest='image_dir', help='Directroy of figure')
    parser.add_argument('-t', '--train', default=False, dest='train', help='Plot training curves or not')

    return parser.parse_args()


if __name__ == '__main__':


    ### To fetch the different args:
    ###     args.files, args.networknames, args.imagename, args.file_dir
    ###     args.image_dir, args.single, args.rows, args.cols


    # Get arguments from the function call
    args = get_args()

    if not len(args.files) == len(args.networknames):
        raise Exception("The number of files and the number of network names are not the same")


    # Append the pandas dataframe and the network names
    dataframes, network_names = [], []
    for i in range(len(args.files)):
        dataframes.append(dl.get_metadata(args.file_dir, args.files[i]))
        network_names.append(args.networknames[i])


    path_name = os.path.join(ABS_PATH, args.image_dir)
    pl.plot_data(dataframes, network_names, path_name, args.imagename, args.metric, train=args.train)
