from argparse import ArgumentParser
import Utils.dataloader as dl
from Utils.show_dataset import multi_slice_viewer


###  COMMANDS  ###
# python show_dataset.py -d test


def get_args():

    ''' Parser used to specify the type of the dataset '''

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset_type', dest='dtype', required=True, \
        help='Dataset type: train, valid, test')

    return parser.parse_args()


if '__name__' == '__name__':

    # Get arguments from the function call
    args = get_args()

    if not args.dtype in ['train', 'valid', 'test']:
        raise Exception('Invalid dtype!')

    data = dl.load_data_as_dict([args.dtype])
    multi_slice_viewer(data[args.dtype]['images'])
