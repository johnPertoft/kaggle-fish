from sklearn.utils import shuffle


def batch_generator(datas, batch_size, forever=True, do_shuffle=True):
    """
    Generator to get batches of supplied data tuples.
    
    datas is a list/tuple of ndarrays that represent the data pairs 
    """
    # TODO: make sure it works if datas is just one ndarray
    N = datas[0].shape[0]
    assert all([d.shape[0] == N for d in datas[1:]])  # Make sure pairs are matched in length

    while True:
        if do_shuffle:
            datas = shuffle(*datas)
        
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            yield [d[i:j] for d in datas]
        
        if not forever:
            break
