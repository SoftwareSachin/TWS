"""
Vector utility functions for similarity calculations and vector operations.
"""

import math
from typing import List, Optional


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score between -1 and 1

    Raises:
        ValueError: If vectors have different lengths
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have the same length: {len(a)} != {len(b)}")

    if not a or not b:
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def cosine_similarity_batch(
    query_vector: List[float], vectors: List[List[float]]
) -> List[float]:
    """
    Calculate cosine similarity between a query vector and multiple vectors efficiently.

    Args:
        query_vector: The query vector to compare against
        vectors: List of vectors to compare with the query vector

    Returns:
        List of cosine similarity scores
    """
    if not vectors:
        return []

    similarities = []
    query_magnitude = math.sqrt(sum(x * x for x in query_vector))

    if query_magnitude == 0:
        return [0.0] * len(vectors)

    for vector in vectors:
        if len(vector) != len(query_vector):
            similarities.append(0.0)
            continue

        dot_product = sum(x * y for x, y in zip(query_vector, vector))
        vector_magnitude = math.sqrt(sum(x * x for x in vector))

        if vector_magnitude == 0:
            similarities.append(0.0)
        else:
            similarities.append(dot_product / (query_magnitude * vector_magnitude))

    return similarities


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Euclidean distance (lower is more similar)
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dot_product(a: List[float], b: List[float]) -> float:
    """
    Calculate dot product of two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Dot product score
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    return sum(x * y for x, y in zip(a, b))


def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize a vector to unit length.

    Args:
        vector: Input vector

    Returns:
        Normalized vector
    """
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return vector

    return [x / magnitude for x in vector]


def top_k_similar(
    query_vector: List[float],
    vectors: List[List[float]],
    k: int = 10,
    threshold: Optional[float] = None,
) -> List[tuple]:
    """
    Find top-k most similar vectors efficiently.

    Args:
        query_vector: The query vector
        vectors: List of vectors to compare against
        k: Number of top results to return
        threshold: Minimum similarity threshold (optional)

    Returns:
        List of (index, similarity_score) tuples, sorted by similarity (highest first)
    """
    similarities = cosine_similarity_batch(query_vector, vectors)

    # Create (index, similarity) pairs
    indexed_similarities = [(i, sim) for i, sim in enumerate(similarities)]

    # Filter by threshold if provided
    if threshold is not None:
        indexed_similarities = [
            (i, sim) for i, sim in indexed_similarities if sim >= threshold
        ]

    # Sort by similarity (descending) and take top k
    indexed_similarities.sort(key=lambda x: x[1], reverse=True)

    return indexed_similarities[:k]


def vector_magnitude(vector: List[float]) -> float:
    """
    Calculate the magnitude (L2 norm) of a vector.

    Args:
        vector: Input vector

    Returns:
        Magnitude of the vector
    """
    return math.sqrt(sum(x * x for x in vector))


def is_valid_vector(vector: List[float], expected_dim: Optional[int] = None) -> bool:
    """
    Check if a vector is valid (non-empty, contains only finite numbers).

    Args:
        vector: Vector to validate
        expected_dim: Expected dimension (optional)

    Returns:
        True if vector is valid, False otherwise
    """
    if not vector:
        return False

    if expected_dim is not None and len(vector) != expected_dim:
        return False

    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector)
