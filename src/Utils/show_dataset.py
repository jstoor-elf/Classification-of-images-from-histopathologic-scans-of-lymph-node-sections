import matplotlib # Fix matplotlib for virtualenv
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np



def multi_slice_viewer(dataset):

    ''' This implementation allows the user to step through a chosen image dataset
        to experience it qualitatively. The images are shown in montages of 64 images
    '''

    remove_keymap_conflicts({'e', 'l'})
    fig, ax = plt.subplots(figsize=(9,3))
    ax.dataset = dataset
    ax.index = 0
    ax.batch_size = 64
    ax.batches = int(np.ceil(dataset.shape[0] / ax.batch_size))
    ax.imshow(montage(image_batch(ax.dataset , ax.index, ax.batch_size, ax.batches)))
    ax.set_title("Image {0}-{1}".format(ax.index*ax.batch_size, (ax.index+1)*ax.batch_size-1))
    ax.axis('off')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()


def next_slice(fig, ax):

    ''' Show the next 64 images in the dataset '''

    ax.index = (ax.index + 1) % ax.batches
    ax.images[0].set_array(montage(image_batch(ax.dataset, ax.index, ax.batch_size, ax.batches)))
    ax.set_title("Image {0}-{1}".format(ax.index*ax.batch_size, (ax.index+1)*ax.batch_size-1))


def previous_slice(fig, ax):

    ''' Show the previous 64 images in the dataset '''

    ax.index = (ax.index - 1) % ax.batches # wrap around using %
    ax.images[0].set_array(montage(image_batch(ax.dataset , ax.index, ax.batch_size, ax.batches)))
    ax.set_title("Image {0}-{1}".format(ax.index*ax.batch_size, (ax.index+1)*ax.batch_size-1))



def process_key(event):

    ''' Process key event. If the key e is pressed the previous 64 images are
        shown. If instead the key l is spressed, the next 64 images are shown.
    '''

    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'e':
        previous_slice(fig, ax)
    elif event.key == 'l':
        next_slice(fig, ax)
    else:
        pass
    fig.canvas.draw()


def remove_keymap_conflicts(new_keys_set):

    ''' Remove any keymap conflicts '''

    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def image_batch(dataset, index, batch_size, batches):

    ''' Get specific images in list of images '''

    return dataset[index * batch_size : (index + 1) * batch_size] \
            if index < batches - 1 else dataset[index * batch_size:]


def montage(X):

    ''' Defines a montage of images '''

    count, m, n, cm = X.shape
    mm = int(np.ceil(np.sqrt(count))) * 2
    nn = mm // 4
    M = np.zeros((nn * m, mm * n, cm), dtype=np.uint8)

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceN:sliceN + n, sliceM:sliceM + m, :] = X[image_id, :, :, :]
            image_id += 1

    return M
