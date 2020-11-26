# Fix gpu configuration if computer has a GPU, mine has not, I set the bool in
# the following conditional to False on my computer
if True:
    import tensorflow as tf
    tf.config.gpu.set_per_process_memory_fraction(0.3)
    tf.config.gpu.set_per_process_memory_growth(True)

from tensorflow.keras.callbacks import History
from Utils import utilities as ul
from Utils.dataloader import load_data_as_dict
from Utils.generator import get_generator
import sys


###  MAIN COMMANDS TO TRAIN THE MODEL ###
# python train.py vgg.json
# python train.py resnet.json
# python train.py densenet.json



if __name__ == '__main__':

    # Check and read in the json file as a dictionary
    args = ul.test_input(sys.argv)

    # Read in the data from the dataloader as dict
    data = load_data_as_dict(['train', 'valid'])

    # Load the model
    model = ul.get_model(data['valid']['images'][0].shape, args["model"], args["modelparams"])
    model.summary()

    # Get the metrics, loss function and optimizer
    metrics, loss_fnc, Optim = ul.get_loss_metrics_optim(args['regimen'])

    # Compile the model
    model.compile(optimizer=Optim(lr=args["regimen"]["lr"]), loss=loss_fnc, \
        metrics=metrics)


    # get training and validation generators, Images aere not normalized!
    args['augmentation']['rescale'] = 1. / 255 # Add rescaling first
    train_gen = get_generator(
        images=data["train"]["images"],
        labels=data["train"]["labels"],
        generator_dict=args['augmentation'],
        batch_size=args["regimen"]['batch_size']
    )

    # Rescale validation images
    data["valid"]["images"] *= 1. / 255


    # fit and append history to csv file using the callback found in Utils.utilities
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=data["train"]["images"].shape[0] // args["regimen"]['batch_size'],
        validation_data=(data["valid"]["images"], data["valid"]["labels"]),
        epochs=args["regimen"]["epochs"],
        verbose=1,
        callbacks = ul.create_callbacks(args["result"])
    )
