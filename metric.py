import faiss
import os
from logging import get_logger

logger = get_logger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

def retrieval_mean_precision(embeddings, labels, k):
    faiss.normalize_L2(embeddings)
    logger.debug(f"Embeddings shape: {embeddings.shape}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _, neighbors = index.search(embeddings, k + 1)
    neighbors = neighbors[:, 1:]
    same_labels = labels[neighbors] == labels[:, None]
    return same_labels.mean().item()

# Todo Develope
def _test_retrieval_mean_precision():
    embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]])
    labels = np.array([0, 1, 2, 0])
    k = 3
    assert retrieval_mean_precision(embeddings, labels, k) == 0.5
    print("Test passed")