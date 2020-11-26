from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.layers import Add, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
import numpy as np



def res_net(input_shape, Base=32, layers_per_block=None, n_classes=1, out_fnc='sigmoid'):


    ''' ResNet implementation

    # Arguments:
        input_shape : shape of input image: (width, height, channels)
        base : number of output features from first convolutional layer
        layers_per_block : list with number of residual blocks in each residual
            module. Each block consists of two or three 1 * 1 convolutions and
            one 3 * 3 convolution.
            - None : Set to ResNet50 ([3, 4, 6, 2])
        n_classes : Number of output classes form network
        out_fnc : Output function of final dense layer
    '''

    # ResNet 50 by default
    if not layers_per_block:
        layers_per_block = [3, 4, 6, 2]


    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Stage 1, always the same
    X = Conv2D(Base, (5, 5), strides = (3, 3), padding='same')(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    # Stage (2-(number_of_modules+1)), this makes the ResNet function generic
    for i in range(len(layers_per_block)):
        out_base = Base * 4

        # Create the blocks for this residual module
        for j in range(layers_per_block[i]):
            s = 1 if (i == 0 and j == 0) or j > 0 else 2
            X = __identity_block(X, f=3, filters=[Base, Base, out_base], s=s, conv1=(j==0))

        Base *= 2


    # AVGPOOL, Output layer
    X = GlobalAveragePooling2D()(X)
    X = Dense(n_classes, activation=out_fnc)(X)

    # Create model
    N = np.sum(np.array(layers_per_block) * 3) + 2
    return Model(inputs = X_input, outputs = X, name='ResNet{}'.format(N))



def __identity_block(X, f, filters, s=2, conv1=False):


    ''' Main identity block implementing the residual block with skip connection

    # Arguments
        X : Input tensor
        f = kernel size for the middlemost convolutional layer
        s = stride size for the first convolutional layer
            - 1 : When it's not the first block of the module
            - 2 : For first residual block in each new residual module
        conv1 : flag for applying 1*1 convolution in the skip connection
        batchnorm: Flag - BatchNormalization after each convolutional layer
    '''


    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value to be used for addition
    X_shortcut = X

    ##### MAIN PATH #####
    # First layer in main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Second layer in main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Third layer in main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = BatchNormalization()(X)

    ##### SHORTCUT PATH ####
    if conv1: # Used if its in the first layer in residual module
        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), \
            padding='valid')(X_shortcut)
        X_shortcut = BatchNormalization()(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
