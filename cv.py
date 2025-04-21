import pandas as pd
from sklearn.model_selection import train_test_split
def split(df: pd.DataFrame, test_size:float=0.2, val_size: float=0.5) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    og_len = len(df['Run1 (ms)'])
    print(f' Len of original dataset: {og_len}')
    ds_train, ds_test = train_test_split(df, test_size = test_size)
    ds_test, ds_val = train_test_split(ds_test, test_size=val_size)

    assert len(ds_train) == 0.8 * og_len
    assert len(ds_val) == 0.1 * og_len
    assert len(ds_test) == 0.1 * og_len

    print('split works')

    return ds_train, ds_val, ds_test

def save(ds: pd.DataFrame, split):
    ds.to_csv(f'Data/SGEMM_{split}.csv', index = False)


if __name__ == "__main__":
    ds = pd.read_csv('sgemm_product.csv')
    print(ds.head())
    train_ds, val_ds, test_ds = split(ds)
    save(train_ds, 'train')
    save(val_ds,  'val')
    save(test_ds,  'test')