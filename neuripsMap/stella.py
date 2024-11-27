from sentence_transformers import SentenceTransformer


class StellaEmbedder:
    def __init__(self, device="cuda"):
        """Initialize the Stella embedder.

        Args:
            device (str): 'cuda' for GPU or 'cpu' for CPU processing
        """
        self.model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5", trust_remote_code=True, device=device
        )
        self.prompt_name = "s2s_query"  # Using s2s for semantic similarity

    def embed_texts(self, texts, batch_size=32):
        """Embed a list of texts.

        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing

        Returns:
            np.ndarray: Array of embeddings with shape (n_texts, 1024)
        """
        # Encode texts using the s2s prompt
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            prompt_name=self.prompt_name,
            show_progress_bar=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )

        return embeddings
