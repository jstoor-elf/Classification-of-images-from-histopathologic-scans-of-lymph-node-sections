from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization
from tensorflow.keras.layers import Dropout, Activation, concatenate, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np


def dense_net(input_shape=None, base=16, dense_blocks=4, blocks_per_layer=None, growth_rate=8, \
        classes=1, out_fnc='sigmoid', dense_depth=5, bottleneck=False, dropout_rate=None):


    ''' DenseNet implementation

    # Arguments
        input_shape : shape of input image: (width, height, channels)
        Base : number of output features after the first convolutional layer
        dense_blocks : Number of dense blocks
        blocks_per_layer : list with number of layers per block
            - None: implement [dense_depth]*dense_blocks
        growth_rate : Number of features added after each convolutional operation
        classes : Number of output classes form network
        out_fnc : Output function of final dense layer
        dense_depth : Layers in each dense block
            - Default : 5, not used if blocks_per_layer is specified
        bottleneck : Flag - applying bottleneck before convolutional layer
        dropout_rate: Dropout rate for dropout layer applied after each fully
            connected layer
            - None: No dropout applied
    '''



    # If number of layers in each block is not given it set to 5 per layer
    if not blocks_per_layer:
        blocks_per_layer = [dense_depth]*dense_blocks
    else:
        if not len(blocks_per_layer) == dense_blocks:
            raise Exception("Number of dense blocks and the length of the blocks per layer have to be the same!")


    # Model input
    X_input = Input(input_shape)

    # This will be used as a counter of the number of channels
    nb_channels = base

    # The denseNet starts with 5*5 convolution with stride 3
    X = Conv2D(nb_channels, (5,5), padding='same', strides=(3,3), use_bias=False)(X_input)

    for i in range(dense_blocks):

        # Dense blocks number of layers specified in dense_blocks[i]
        X = dense_block(X, blocks_per_layer[i], growth_rate, dropout_rate, bottleneck)

        # Add a transition block if we're not at the end
        if (dense_blocks - 1) > i:
            nb_channels += growth_rate*blocks_per_layer[i]
            X = transition_layer(X, nb_channels, dropout_rate=dropout_rate)


    # Output layer
    X = output_layer(X)
    X_out = Dense(classes, activation=out_fnc, use_bias=False)(X)

    N = np.sum((1+bottleneck) * np.asarray(blocks_per_layer)) + len(blocks_per_layer) + 1
    return Model(inputs = X_input, outputs =  X_out, name = 'DenseNet{}'.format(N))



def dense_block(X, depth=5, base=12, dropout_rate=None, bottleneck=False):

    ''' Dense block

    # Arguments:
        X = Input tensor
        base = number of output feuture maps after each convolutional operation
        dropout_rate = dropout rate for the dropout applied after each
            bottleneck layer
            - None : No dropout is applied
        bottleneck : Flag - use bottleneck layer before convolutional layer
    '''

    # Future inputs will be concatenated to this list
    for i in range(depth):
        X_n = composition_layer(X, base, dropout_rate=dropout_rate)
        X = concatenate([X, X_n], axis=-1)

    return X



def bottleneck_layer(X, base, dropout_rate=None):

    ''' Bottleneck layer: reduce complexity within dense block

    # Arguments:
        X = Input tensor
        base = 4*base is the number of output feutures after convolutional layer
        dropout_rate = rate for dropout applied after each bottleneck layer
            - None : No dropout applied
    '''

    # Perform BN -> ReLU -> 1*1 Conv2D
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = Conv2D(base * 4, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(X)

    # Adding dropout
    if dropout_rate:
            X = Dropout(dropout_rate)(X)
    return X



def composition_layer(X, base, dropout_rate=None, bottleneck=False):


    ''' Composition layer: reduce complexity within dense block. Consisting of
        conditional bottleneck layer followed by BN -> ReLU -> Conv2D

    # Arguments:
        X = Input tensor
        base = number of output feutures after convolutional layer
        dropout_rate = rate for dropout applied after each bottleneck layer
            - None : No dropout applied
        bottleneck : Flag - use bottleneck layer before convolutional layer
    '''


    # If bottleneck is used, i.e. it's used to reduce the model complexity and size,
    if bottleneck:
        bottleneck_layer(X, base, dropout_rate=dropout_rate)

    # Perform BN -> ReLU -> Conv2D
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    return Conv2D(base, kernel_size=(3, 3), padding ='same', use_bias=False)(X)



def transition_layer(X, base, dropout_rate=None):

    ''' Transition layer: to be applied between dense blocks. Performs BN ->
        ReLU -> Conv2D

    # Arguments:
        X = Input tensor
        base = number of output feutures after convolutional layer
        dropout_rate = rate for dropout applied after each bottleneck layer
            - None : No dropout applied
    '''

    # Perform BN -> ReLU -> Conv2D
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(base, kernel_size=(1,1), padding='same', use_bias=False)(X)

    # Adding dropout
    if dropout_rate:
        X = Dropout(dropout_rate)(X)
    return AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)



def output_layer(X):

    ''' Output layer. Performs BN -> ReLU -> GlobalAvaragePooling

    # Arguments:
        X = Input tensor
    '''

    # Perform BN -> ReLU -> GlobalAvaragePooling
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    return GlobalAveragePooling2D()(X)
