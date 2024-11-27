import time

from sklearn.manifold import TSNE


class EmbeddingReducer:
    def __init__(self, n_components=2, perplexity=30, n_iter=1000, random_state=42):
        """Initialize the t-SNE reducer.

        Args:
            n_components (int): Number of dimensions to reduce to (default: 2)
            perplexity (float): t-SNE perplexity parameter (default: 30)
            n_iter (int): Number of iterations for optimization (default: 1000)
            random_state (int): Random seed for reproducibility
        """
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            init="pca",  # Initialize with PCA for better results
            learning_rate="auto",  # Automatic learning rate selection
            n_jobs=-1,  # Use all available cores
        )

    def reduce_embeddings(self, embeddings):
        """Reduce dimensionality of embeddings using t-SNE.

        Args:
            embeddings (np.ndarray): Array of shape (n_samples, n_features)

        Returns:
            np.ndarray: Reduced embeddings of shape (n_samples, n_components)
        """
        print("Starting t-SNE reduction...")
        start_time = time.time()

        reduced_embeddings = self.tsne.fit_transform(embeddings)

        print(f"t-SNE reduction completed in {time.time() - start_time:.2f} seconds")
        return reduced_embeddings
