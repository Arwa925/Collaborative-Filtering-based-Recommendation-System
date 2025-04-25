from surprise import accuracy

def calculate_rmse(predictions):
    """
    Calculate RMSE from predictions.
    """
    return accuracy.rmse(predictions)

def precision_recall_at_k(predictions, k=10):
    """
    Calculate Precision@K and Recall@K.
    """
    top_k = []
    for uid, iid, true_r, est, _ in predictions:
        top_k.append((uid, iid, true_r, est))
    
    top_k.sort(key=lambda x: x[3], reverse=True)
    
    relevant = [x for x in top_k if x[2] >= 4]
    retrieved = [x for x in top_k[:k]]
    
    precision = len(set([x[1] for x in relevant]) & set([x[1] for x in retrieved])) / k
    recall = len(set([x[1] for x in relevant]) & set([x[1] for x in retrieved])) / len(relevant)
    
    return precision, recall
