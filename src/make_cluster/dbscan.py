import logging
from datetime import datetime

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import umap.umap_ as umap

import warnings
warnings.filterwarnings("ignore")

from log import setup_logging

random_state = 32


def clusterization(embeddings_path, result_path):
    
    embeddings = pd.read_csv(embeddings_path, index_col=0)
    logging.info(f"Input features shape {embeddings.shape}")
    
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(embeddings)
    logging.info(f"Features shape after standart scaling {dataset_scaled.shape}")

    emb = umap.UMAP(random_state=12).fit(dataset_scaled)
    umap_embedding = emb.embedding_

    results = pd.DataFrame(
                            {'x': umap_embedding[:,0],
                            'y':  umap_embedding[:,1],
                            })

    logging.info(f"Features shape after UMAP {results.shape}")

    dbscan_clusterer = DBSCAN(eps=0.9, min_samples=150).fit(umap_embedding)
    logging.info(f'DBSCAN params: eps 0.9, min_samples 150')
    
    embeddings = embeddings.reset_index()
  
    embeddings["dbscan_preds"] = dbscan_clusterer.labels_
    logging.info(f"Count unique clusters {len(set(dbscan_clusterer.labels_))}")
    
    embeddings[["CLIENT_IP", "dbscan_preds"]].to_csv(result_path, index=False)
    logging.info(f"Predictions saved")


        
if __name__ == "__main__":
    setup_logging()
    start = datetime.now()
    logging.info(f"Start clusterization at {start}")
    clusterization(embeddings_path='../../data/embeddings.csv', result_path='../../data/result.csv')
    logging.info(f"Clusterization finished in {datetime.now() - start} seconds")