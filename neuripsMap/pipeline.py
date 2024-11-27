import os

import hdbscan
import numpy as np
import pandas as pd
from info_extractor import extract_papers
from stella import StellaEmbedder
from tsne import EmbeddingReducer


class NeurIPSMapper:
    def __init__(
        self,
        path: str,
        authors_of_interest: list = [],
        titles_crossref: list = [],
        embedder: bool = False,
    ):
        self.path = path
        self.authors_of_interest = authors_of_interest
        self.titles_crossref = titles_crossref

        if embedder:  # Initialize embedder : means the user wants to perform the whole pipeline without providing embeddings
            self.embedder = StellaEmbedder(device="mps")

        self.reducer = EmbeddingReducer(
            n_components=2,
            perplexity=50,  # Higher for larger datasets
            n_iter=2000,  # More iterations for better convergence
            random_state=42,
        )

    def extract_papers(self):
        raw = r""
        for file in os.listdir(self.path):
            if file.startswith("session"):
                with open(os.path.join(self.path, file), "r") as f:
                    raw += f.read()

        df = extract_papers(raw, self.authors_of_interest, self.titles_crossref)
        self.df = df

        return df

    def embed_texts(self, df: pd.DataFrame):
        abstracts = [
            f"# {row['title']}\n\n{row['abstract']}" for _, row in df.iterrows()
        ]
        embeddings = embedder.embed_texts(abstracts)

        return embeddings

    def load_embeddings(self, path: str):
        self.sentence_embeddings = np.load(path, allow_pickle=True)
        print(f"Embeddings loaded from {path}")
        return self.sentence_embeddings


if __name__ == "__main__":
    mapper = NeurIPSMapper(path="data", authors_of_interest=[], embedder=False)
    embeddings = mapper.load_embeddings("neuripsMap/abstracts_embeddings.npy")
    print(f"Embeddings shape: {embeddings.shape}")
    try:
        # load local reduced embeddings if available
        reduced_embeddings = np.load("neuripsMap/tsne_embeddings.npz")[
            "reduced_embeddings"
        ]
        print("Reduced embeddings loaded from file")
    except:
        print("Reduced embeddings not found, reducing embeddings...")
        reduced_embeddings = mapper.reducer.reduce_embeddings(embeddings)
        # Save results if path provided
        save_path = "neuripsMap/tsne_embeddings.npz"
        np.savez(
            save_path, embeddings=embeddings, reduced_embeddings=reduced_embeddings
        )
        print(f"Results saved to {save_path}")

    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

    # Cluster embeddings with HDBSCAN
    # Initialize HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        gen_min_span_tree=True,  # Enable hierarchy visualization
        prediction_data=True,  # Enable soft clustering
    )

    example = np.load(
        "/Users/joanvelja/Downloads/arxiv_ml_data_map.npy", allow_pickle=True
    )

    # Perform clustering
    print("Performing HDBSCAN clustering...")
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    probabilities = clusterer.probabilities_

    # Get cluster persistence scores
    persistence_scores = clusterer.cluster_persistence_

    # Calculate outlier scores
    outlier_scores = clusterer.outlier_scores_

    dict_results = {
        "embeddings": embeddings,
        "reduced_embeddings": reduced_embeddings,
        "cluster_labels": cluster_labels,
        "probabilities": probabilities,
        "persistence_scores": persistence_scores,
        "outlier_scores": outlier_scores,
        "clusterer": clusterer,  # For accessing hierarchy later
    }

    # Plot the clusters (coloring the points by the cluster)
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hue=cluster_labels,
        palette="tab10",
        legend="full",
    )

    plt.title("HDBSCAN Clustering of NeurIPS Abstracts")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Cluster")

    plt.tight_layout()
    plt.show()
