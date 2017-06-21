import numpy as np

def make_one_hot(target,labels):
    targets = np.zeros((len(target),labels))
    targets[np.arange(len(target)),target - 1] = 1
    return targets


def get_data(datafile,seed=21,type=np.float32,delim=','):
    data   = np.loadtxt(datafile,delimiter=delim,dtype=type)
    print('loaded data')
    print data.shape

    np.random.seed(seed)
    p = np.random.permutation(data.shape[0])
    data = data[p]
    return data


def reorganize(X,timesteps=1):
    sample_size = X.shape[0]
    total_features = X.shape[1]
    removal = (total_features%timesteps)
    fixed_features = (total_features - removal)
    step_size = fixed_features / timesteps
    return X[:,:fixed_features].reshape(sample_size,timesteps,step_size)
