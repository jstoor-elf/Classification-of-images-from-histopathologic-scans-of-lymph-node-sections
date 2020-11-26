
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_generator(images, labels, generator_dict, batch_size):

    ''' Custom generator that recieves a dict of augmentations to be applied to
        the image.
    '''

    generator = ImageDataGenerator(**generator_dict)
    return __my_generator(generator, images, labels, batch_size)


# Custom generator to apply transformation to the image
def __my_generator(gen, X, Y, batch_size, seed=1):
    genX = gen.flow(X, Y, batch_size=batch_size, seed=seed)
    while True:
        XX = genX.next() # Create image batch, masks are not augmented
        yield XX[0], XX[1]
