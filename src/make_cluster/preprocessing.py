import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from transformers import RobertaTokenizer

from log import setup_logging

import warnings
warnings.filterwarnings("ignore")


def get_token(x, tokenizer):
    """возвращает эмбеддинг текста"""
    return tokenizer(x, add_special_tokens=False)['input_ids']


def handle_features(load_path, upload_path, pretrained_model='sberbank-ai/ruRoberta-large'):
    
    dataset = pd.read_csv(load_path)
    dataset = dataset.set_index('CLIENT_IP')
    logging.info(f"Source dataset shape {dataset.shape}")
    
    dataset = dataset.drop(index=dataset[dataset.duplicated(keep=False)].index)
    logging.info(f"Dataset shape after remove duplicates {dataset.shape}")

    dataset['var_name_eql_value'] = np.where(dataset['MATCHED_VARIABLE_NAME'] == dataset['MATCHED_VARIABLE_VALUE'], 1, 0 )
    dataset['variable_src_in_name'] = dataset.apply(lambda x: str(x.MATCHED_VARIABLE_SRC) in str(x.MATCHED_VARIABLE_NAME), axis=1) * 1

    dataset['bad_useragent'] = np.where(dataset['CLIENT_USERAGENT'].isna(), 1, 0)
    dataset['bad_name'] = np.where(dataset['MATCHED_VARIABLE_NAME'].isna(), 1, 0)
    dataset['bad_value'] = np.where(dataset['MATCHED_VARIABLE_VALUE'].isna(), 1, 0)
    dataset['bad_src'] = np.where(dataset['MATCHED_VARIABLE_SRC'].isna(), 1, 0)
    dataset['unknown_event'] = np.where(dataset['EVENT_ID'].isna(), 1, 0)

    dataset['REQUEST_SIZE'] = pd.to_numeric(dataset['REQUEST_SIZE'], errors='coerce')
    dataset['bad_req_size'] = np.where(dataset['REQUEST_SIZE'].isna(), 1, 0)
    dataset['REQUEST_SIZE'].fillna(-1, inplace=True)
    dataset['REQUEST_SIZE'] = dataset['REQUEST_SIZE'].astype('int')

    dataset['RESPONSE_CODE'] = pd.to_numeric(dataset['RESPONSE_CODE'], errors='coerce')
    dataset['bad_resp_code'] = np.where(dataset['RESPONSE_CODE'].isna(), 1, 0)
    dataset['RESPONSE_CODE'].fillna(-1, inplace=True)
    dataset['RESPONSE_CODE'] = dataset['RESPONSE_CODE'].astype('int')

    dataset = dataset.fillna('Неизвестно')
    dataset.drop(['EVENT_ID'], axis=1, inplace=True)
    logging.info(f"NULL values in dataset: {dataset.isnull().sum().sum()}")
    
    cat_features = list(dataset.loc[:, dataset.dtypes == object].columns)
    num_features = list(set(dataset.columns).difference(set(cat_features)))
    
    logging.info(f"Len of cat_features {len(cat_features)} \n, len of num_features {len(num_features)}")

    for col in cat_features:
        dataset[col] = dataset[col].astype('str')
        
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
        
    for col in cat_features:
        dataset[col + '_median'] = dataset[col].apply(lambda x: np.median(get_token(x, tokenizer)))
        dataset[col + '_sum'] = dataset[col].apply(lambda x: np.sum(get_token(x, tokenizer)))
        dataset[col + '_mean'] = dataset[col].apply(lambda x: np.mean(get_token(x, tokenizer)))
        dataset[col + '_mode'] = dataset[col].apply(lambda x: stats.mode(get_token(x, tokenizer)).mode[0])
        dataset[col + '_min'] = dataset[col].apply(lambda x: np.min(get_token(x, tokenizer)))
        dataset[col + '_max'] = dataset[col].apply(lambda x: np.max(get_token(x, tokenizer)))
        dataset[col + '_std'] = dataset[col].apply(lambda x: np.std(get_token(x, tokenizer)))
        
    for col in cat_features:
        dataset[col + '_count'] = dataset[col].map(dataset.groupby(col).size())

    num_features = list(set(dataset.columns).difference(set(cat_features)))
    
    logging.info(f"New feature's shape {dataset[num_features].shape}")

    dataset[num_features].to_csv(upload_path, index=True)
    logging.info(f"New features saved to {upload_path}")



if __name__ == "__main__":
    setup_logging()
    start = datetime.now()
    logging.info(f"Start features processing at {start}")
    handle_features(load_path='../../data/part_10.csv', upload_path='../../data/embeddings.csv')
    logging.info(f"Features processed in {datetime.now() - start} seconds")