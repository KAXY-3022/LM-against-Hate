from tqdm import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool

def batchify(data, batch_size, description="Processing"):
    """
    Yield successive n-sized batches from data, with a progress bar.

    Args:
        data: List or iterable to batch.
        batch_size: Size of each batch.
        description: Description for the progress bar.

    Yields:
        Batches of data of size batch_size.
    """
    total = len(data)
    for i in tqdm(range(0, total, batch_size), desc=description, leave=True):
        yield data[i:i + batch_size]
        
        
def parallel_process(df, fn_to_execute, num_cores=4):
    # create a pool for multiprocessing
    pool = Pool(num_cores)

    # split your dataframe to execute on these pools
    splitted_df = np.array_split(df, num_cores)

    # execute in parallel:
    split_df_results = pool.map(fn_to_execute, splitted_df)

    # combine your results
    df = pd.concat(split_df_results)

    pool.close()
    pool.join()
    return df
