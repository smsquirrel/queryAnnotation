import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def calculate_mrr(scores, relevant_indices, K=1000):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a set of queries and documents,
    considering only the top K documents.

    :param scores: A 2D numpy array of shape (n_queries, n_documents) representing the score matrix.
    :param relevant_indices: A list of lists, where each sublist contains the indices of relevant documents for the corresponding query.
    :param K: The number of top documents to consider when looking for relevant documents.
    :return: The MRR score.
    """
    reciprocal_ranks = []
    
    for query_idx, relevant_docs in enumerate(relevant_indices):
        query_scores = scores[query_idx]
        ranked_docs = np.argsort(-query_scores)[:K]  # Only consider the top K documents
        # Find the rank of the highest ranked relevant document within the top K documents
        relevant_ranks = [np.where(ranked_docs == rel_doc)[0][0] for rel_doc in relevant_docs if rel_doc in ranked_docs]

        if relevant_ranks:
            highest_rank = min(relevant_ranks) + 1  # Add 1 because ranks start from 1, not 0
            reciprocal_ranks.append(1.0 / highest_rank)
        else:
            # If no relevant document is found within the top K, append 0 to reciprocal_ranks
            reciprocal_ranks.append(0.0)
    
    # compute MRR
    mrr = np.mean(reciprocal_ranks)
    return mrr