import tensorflow as tf
import pandas as pd

from config import CLASS_NAMES
from util import batch_generator

result = pd.DataFrame(test_res, columns=CLASS_NAMES)
result.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
result.to_csv('submission.csv', index=False)


def infer(sess, model, X):
    # TODO: assert that current graph represents the model?

    batches = batch_generator([X], batch_size=128, forever=False, do_shuffle=False) 
    probs = []
    for batch in batches:
        batch = batch[0]
        probs.append(sess.run(model.softmax, feed_dict={X: batch, model.keep_prob: 1.0}))
    return probs

if __name__ == "__main__":
    pass
