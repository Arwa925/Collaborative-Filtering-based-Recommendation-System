from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise import accuracy
from surprise.model_selection import train_test_split

def user_based_collaborative_filtering(data):
    """
    Implement User-Based Collaborative Filtering.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data[['user_id', 'movie_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    
    sim_options = {
        'name': 'cosine',
        'user_based': True  # User-based collaborative filtering
    }
    
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    return predictions

def item_based_collaborative_filtering(data):
    """
    Implement Item-Based Collaborative Filtering.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data[['user_id', 'movie_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    
    sim_options = {
        'name': 'cosine',
        'user_based': False  # Item-based collaborative filtering
    }
    
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    return predictions
