def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    pre_count = 0
    
    for i in range(k):
        if recommended[i] in relevant:
            pre_count += 1
    return [pre_count / k, pre_count / len(relevant)]