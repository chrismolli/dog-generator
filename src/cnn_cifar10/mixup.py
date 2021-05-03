import numpy as np

def mixup_extend_data(x,y,n):
    """ MIXUP_EXTEND_DATA will use the mixup technique to append
        n inter-class representations to the given data. y must be
        in a one hot representation.
    """
    # copy data
    x_extend = []
    y_extend = []

    # create new data
    for i in range(n):
        # draw two indices
        first = int(x.shape[0] * np.random.rand())
        second = int(x.shape[0] * np.random.rand())
        while second is first:
            second = int(np.round(x.shape[0] * np.random.rand(), 0))
        # draw mixup ratio from [0.2,0.4]
        mix_ratio = 0.2 * (np.random.rand() + 1)
        # mix up
        (x_, y_) = mixup(x[first],x[second],y[first],y[second],mix_ratio)
        # append to extended data set
        x_extend.append(x_)
        y_extend.append(y_)

    # join datasets
    x_extend = np.stack(x_extend,axis=0)
    x_extend = np.concatenate([x,x_extend],axis=0)
    y_extend = np.stack(y_extend, axis=0)
    y_extend = np.concatenate([y, y_extend], axis=0)

    # return modified dataset
    return x_extend, y_extend

def mixup(x1,x2,y1,y2,mix_ratio):
    """ MIXUP creates a inter-class datapoint using mix_ratio
    """
    x = mix_ratio * x1 + (1-mix_ratio) * x2
    y = mix_ratio * y1 + (1-mix_ratio) * y2
    return (x,y)
