import numpy as np
import torch
from chromadb.api.types import Documents, EmbeddingFunction
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_length: int = 512
    ):
        # Check if MPS is available (Apple Silicon)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        if self.device == 'cpu':
            logger.warning("MPS/GPU is not available. Using CPU instead.")
        
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length

        if self.device == 'mps':
            logger.info("Using Apple Silicon GPU (MPS)")

    def __call__(self, texts: Documents) -> Embeddings:
        if not texts:
            return []  # Return empty list instead of np.array([])

        try:
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                try:
                    with torch.amp.autocast("mps"):
                        embeddings = self.model.encode(
                            batch,
                            batch_size=self.batch_size,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            max_length=self.max_length
                        )
                        # Convert numpy array to list of lists
                        embeddings = embeddings.tolist()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        logger.warning(f"OOM error, retrying with batch size {self.batch_size // 2}")
                        self.batch_size = self.batch_size // 2
                        embeddings = self.model.encode(
                            batch,
                            batch_size=self.batch_size,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            max_length=self.max_length
                        ).tolist()
                    else:
                        raise e
                
                all_embeddings.extend(embeddings)  # Use extend instead of append
                
                if i % (self.batch_size * 10) == 0:
                    torch.cuda.empty_cache()

            return all_embeddings  # This is now List[List[float]], matching Embeddings type
        except Exception as e:
            logger.error(f"Error in embedding generation: {str(e)}")
            raise