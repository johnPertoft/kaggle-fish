import tensorflow as tf
import pandas as pd

from config import CLASS_NAMES

result = pd.DataFrame(test_res, columns=CLASS_NAMES)
result.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
result.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    pass
