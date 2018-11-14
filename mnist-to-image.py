import os
import numpy as np
from PIL import Image

def load_mnist():
    data_dir = './data'
    dataset_name = 'mnist'

    data_dir = os.path.join(data_dir, dataset_name)

    fd = open(os.path.join(data_dir ,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd ,dtype=np.uint8)
    trX = loaded[16:].reshape((60000 ,28 ,28 ,1)).astype(np.float)


    for i in range(trX.shape[0]):
        a = trX[i,:,:,0].astype(np.uint8)
        img = Image.fromarray(a, mode='L')
        img.save('data/mnist2/tr{}.png'.format(i))

    # fd = open(os.path.join(data_dir ,'train-labels-idx1-ubyte'))
    # loaded = np.fromfile(file=fd ,dtype=np.uint8)
    # trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir ,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd ,dtype=np.uint8)
    teX = loaded[16:].reshape((10000 ,28 ,28 ,1)).astype(np.float)

    for i in range(teX.shape[0]):
        a = teX[i, :, :, 0].astype(np.uint8)
        img = Image.fromarray(a, mode='L')
        img.save('data/mnist2/te{}.png'.format(i))
        break

    # fd = open(os.path.join(data_dir ,'t10k-labels-idx1-ubyte'))
    # loaded = np.fromfile(file=fd ,dtype=np.uint8)
    # teY = loaded[8:].reshape((10000)).astype(np.float)

    # trY = np.asarray(trY)
    # teY = np.asarray(teY)

    # X = np.concatenate((trX, teX), axis=0)
    # y = np.concatenate((trY, teY), axis=0).astype(np.int)

    # seed = 547
    # np.random.seed(seed)
    # np.random.shuffle(X)
    # np.random.seed(seed)
    # np.random.shuffle(y)

    # y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    # for i, label in enumerate(y):
    #     y_vec[i ,y[i]] = 1.0

    return # X/ 255., y_vec

if __name__ == "__main__":
    load_mnist()
