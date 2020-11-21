import numpy as np
import torch
from config import *

def load_mapper(name, gpu=True):
    if "ota" in name:
        return PytorchMapper.load(name)
    if "mlp" in name:
        return MLPMapper.load(name)
    if "linear" in name:
        return LinearMapper.load(name)
    if "ortho" in name:
        return OrthogonalMapper.load(name)
    raise Exception(f"Cannot infer mapper class from {name}")

class Mapper:
    pass


class PytorchMapper(Mapper):
    def train(self):
        raise NotImplementedError

    def apply(self, x, verbose=0):
        expanded = False
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
            expanded = True

        y = []
        batchsize = 512
        param = next(self.model.parameters())
        for i in range(0, x.shape[0], batchsize):
            batch = torch.tensor(x[i:i+batchsize]).to(dtype = param.dtype, device = param.device)
            y.append(self.model(batch).detach().cpu().numpy())

        y = np.concatenate(y, axis = 0)
        if expanded:
            y = y.squeeze(0)

        return y

    @classmethod
    def load(cls, path):
        import torch
        obj = cls()
        if not path.endswith(".pt"):
            path += ".pt"
        if not os.path.exists(path):
            path = os.path.join(MAPPERS_DIR, path)
        obj.model = torch.load(path)
        if torch.cuda.is_available():
            obj.model = obj.model.cuda()
        obj.model.eval()

        return obj

class LinearMapper(Mapper):
    def train(self, x, y, w=None, verbose=0):
        if not w is None:
            w_sqrt = np.expand_dims(np.sqrt(w), -1)
            x *= w_sqrt
            y *= w_sqrt

        self.model = np.linalg.lstsq(x, y, rcond=None)[0]

    def apply(self, x, verbose=0):
        return x.dot(self.model)

    def save(self, path):
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, self.model)

    @classmethod
    def load(cls, path):
        obj = cls()
        if not path.endswith(".npy"):
            path += ".npy"

        if not os.path.exists(path):
            path = os.path.join(MAPPERS_DIR, path)

        obj.model = np.load(path)
        return obj

class OrthogonalMapper(LinearMapper):
    def train(self, x, y, w=None, verbose=0):
        from scipy.linalg import orthogonal_procrustes
        if not w is None:
            w_sqrt = np.expand_dims(np.sqrt(w), -1)
            x *= w_sqrt
            y *= w_sqrt

        if x.shape[-1] < y.shape[-1]:
            diff = y.shape[-1] - x.shape[-1]
            zeros = np.zeros_like(x[...,0])
            zeros = np.stack([zeros] * diff, -1)
            x = np.concatenate((x, zeros), -1)
        
        self.model = orthogonal_procrustes(x, y)[0]
        
    def apply(self, x, verbose=0):
        if x.shape[-1] < self.model.shape[0]:
            diff = self.model.shape[0] - x.shape[-1]
            zeros = np.zeros_like(x[...,0])
            zeros = np.stack([zeros] * diff, -1)
            x = np.concatenate((x, zeros), -1)

        return x.dot(self.model)


class MLPMapper(Mapper):
    def __init__(self):
        super(MLPMapper, self).__init__()
        import keras.backend as K
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(tf.Session(config=config))
    
    @staticmethod
    def make_generator(x, y, w, seed, batchsize):
        state = np.random.RandomState(seed)
        idx = list(range(x.shape[0]))
        current_idx = []

        if w is None:
            w = np.ones((x.shape[0],))

        while True:
            state.shuffle(idx)
            for i in idx:
                current_idx.append(i)
                if len(current_idx) == batchsize:
                    yield(x[current_idx], y[current_idx], w[current_idx])
                    current_idx.clear()

    def train(self, x, y, 
            hidden_sizes, steps, batchsize, activation,
            optimizer, loss, w=None, seed=0, verbose=0):

        from keras.models import Sequential
        from keras.layers import Dense     
        from keras.initializers import glorot_uniform
        
        self.model = Sequential()
        initializer = glorot_uniform(seed)
        
        input_shape = (x.shape[-1],)
        for units in hidden_sizes:
            self.model.add(\
                    Dense(input_shape=input_shape, units=units, activation = activation, 
                        kernel_initializer = initializer))
            input_shape = (units,)

        self.model.add(\
                Dense(units=y.shape[-1], activation="linear", kernel_initializer=initializer))

        self.model.compile(loss = loss, optimizer = optimizer)
        generator = self.make_generator(x, y, w, seed, batchsize = batchsize)
        self.model.fit_generator(generator, steps_per_epoch=steps, verbose=verbose)

    def apply(self, x, verbose=0):
        if len(x.shape) == 1:
            return self.model.predict_on_batch(np.expand_dims(x, 0)).squeeze(0)
        return self.model.predict(x, batch_size = 1000, verbose = verbose)

    def save(self, path):
        if not path.endswith(".hdf5"):
            path += ".hdf5"
        self.model.save(path)
    
    @classmethod
    def load(cls, path):
        from keras.models import load_model
        
        obj = cls()
        if not path.endswith(".hdf5"):
            path += ".hdf5"
        
        if not os.path.exists(path):
            path = os.path.join(MAPPERS_DIR, path)

        obj.model = load_model(path)

        return obj
