from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np


def vgg_net(input_shape, Base=32, basefactors=None, layers=None, fulllayers=None, \
            maxpool=None, out_c=1, out_fnc="sigmoid", batchnorm2D=False, dropout=None):


    ''' VGG implementation

    # Arguments:
        input_shape : shape of input image: (width, height, channels)
        Base : number of output features to proceed from
        basefactors : list holding a factor for each block to multiply the
            the Base with to get number of output features for that block
            - None: defaults to [1, 2, 4, 8, 8]
        layers : number of convolutional layers in each block
            - None: defaults to [2, 2, 3, 3, 3]
        not fulllayers: list holding who's length determines the number of
            fully connected layers and their number of neurons
            - None: defaults to [128, 64]
        batchnorm2D : flag denoting Batch-normalization after each convolutional
            layer
        maxpool : list with flags denoting if the block ends with 2*2 maxpooling
            - None: set to true for all blocks
        n_classes : Number of output classes form network
        out_fnc : Output function of final dense layer
        batchnorm2D: Flag - BatchNormalization after each convolutional layer
        dropout: Dropout rate for dropout layer applied after each fully
            connected layer
            - Default: None, no dropout applied
    '''


    # Implement vgg16 if nothing is specified
    if not basefactors:
        basefactors = [1, 2, 4, 8, 8]
    if not layers:
        layers = [2, 2, 3, 3, 3]
    if not fulllayers:
        fulllayers = [128, 64]
    if not maxpool:
        maxpool = [True, True, True, True, True]


    # Store modules in sequential container
    X_input = Input(input_shape)
    X = X_input

    # Produce convolutional blocks
    for i in range(len(basefactors)):
        # Construct each block, using indivdual layers
        X = __vgg_block(X, Base * basefactors[i], layers[i], batchnorm2D, maxpool[i])


    # Construct fully convolutinal layer
    X = Flatten()(X)
    for i in range(len(fulllayers)):
        X = Dense(fulllayers[i], activation='relu')(X)

        # If droput
        if dropout:
            X = Dropout(rate=dropout)(X)

    X_out = Dense(out_c, activation=out_fnc)(X)

    N = np.sum(layers) + len(fulllayers) + 1
    return Model(inputs = X_input, outputs =  X_out, name = 'VGG{}'.format(N))



def __vgg_block(X, fact, n_layers, batchnorm2D, maxpool):

    ''' VGG block

    # Arguments:
        X : input tensor
        fact : number of output features from each convolution
        n_layers : number of convolutional layers in block
        batchnorm2D : flag denoting if Batch-normalization is going to be
            applied after each convolutional layer
        maxpool : flag denoting if the block ends with maxpooling
    '''

    # Construct each block, using indivdual layers
    for n in range(n_layers):
        X = Conv2D(fact, (3, 3), padding='same')(X)

        # If batchnormalization
        if batchnorm2D:
            X = BatchNormalization()(X)

        # Activation after conditional batch normalization
        X= Activation('relu')(X)

    # Maxpooling before the end of each block
    if maxpool:
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    return X
