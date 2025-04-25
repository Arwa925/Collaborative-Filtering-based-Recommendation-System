import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load data from a CSV file and convert it to a DataFrame.
    """
    data = pd.read_csv(file_path, delimiter='\t', header=None)
    return data

def load_user_data(file_path):
    """
    Load user data (e.g., age, gender, occupation).
    """
    user_data = pd.read_csv(file_path, delimiter='|', header=None, encoding='latin-1')
    user_data.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    return user_data

def load_movie_data(file_path):
    """
    Load movie data (e.g., title, genres).
    """
    movie_data = pd.read_csv(file_path, delimiter='|', header=None, encoding='latin-1')
    movie_data.columns = ['movie_id', 'title', 'genres']
    return movie_data

def prepare_data(ratings_path, movies_path, users_path):
    """
    Load and merge ratings, movie, and user data.
    """
    ratings = load_data(ratings_path)
    users = load_user_data(users_path)
    movies = load_movie_data(movies_path)

    ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

    ratings_with_user_data = pd.merge(ratings, users, on='user_id')
    full_data = pd.merge(ratings_with_user_data, movies, on='movie_id')

    return full_data
