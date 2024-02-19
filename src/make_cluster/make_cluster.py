import logging
from datetime import datetime

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import umap.umap_ as umap

import warnings
warnings.filterwarnings("ignore")

from log import setup_logging

random_state = 32


def clusterization(embeddings_path, result_path, num_clusters = 45):
    embeddings = pd.read_csv(embeddings_path)
    
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(embeddings)

    emb = umap.UMAP(random_state=12).fit(dataset_scaled)
    umap_embedding = emb.embedding_


    results = pd.DataFrame(
                            {'x': umap_embedding[:,0],
                            'y':  umap_embedding[:,1],
                            })


    kmeans_clusterer = KMeans(n_clusters=num_clusters, init='k-means++',
                            verbose=0, random_state=random_state,
                            algorithm='auto').fit(umap_embedding)

    embeddings["kmeans_preds"] = kmeans_clusterer.labels_
    
    embeddings["kmeans_preds"].to_csv(result_path, index=False)
    
    print(embeddings["kmeans_preds"])
        
if __name__ == "__main__":
    setup_logging()
    start = datetime.now()
    logging.info(f"Start clusterization at {start}")
    clusterization(embeddings_path='../../data/embeddings.csv', result_path='../../data/result.csv')
    logging.info(f"Clusterization finished in {datetime.now() - start} seconds")