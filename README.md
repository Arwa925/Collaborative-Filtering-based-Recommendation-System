
# Collaborative Filtering Recommender System

This project builds a Collaborative Filtering-based Recommendation System that suggests movies to users based on their past interactions and preferences, and the preferences of similar users. Collaborative filtering assumes that users with similar behaviors will enjoy similar items.

### Requirements
- Python 3.x
- Libraries:
    - pandas
    - numpy
    - scikit-learn
    - surprise (for collaborative filtering algorithms)
    - matplotlib / seaborn (for visualizations)
- MovieLens 100K Dataset: [MovieLens 100K Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/collaborative-filtering-recommender.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### File Structure:
- `data/`: Contains the MovieLens dataset files.
- `notebooks/`: Jupyter Notebooks for preprocessing, user-based and item-based CF, and evaluation.
- `src/`: Python scripts for loading data, collaborative filtering logic, and evaluation metrics.
- `results/`: Contains the evaluation metrics and top-N movie recommendations.

### Usage
1. Load and preprocess the data:
    - Run `01_data_preprocessing.ipynb`
2. Implement User-based Collaborative Filtering:
    - Run `02_user_based_cf.ipynb`
3. Implement Item-based Collaborative Filtering:
    - Run `03_item_based_cf.ipynb`
4. Evaluate and visualize the model:
    - Run `04_evaluation_and_visualization.ipynb`

### Results
- Top-N recommendations saved in `results/top_n_recommendations.csv`.
- Evaluation metrics saved in `results/evaluation_metrics.json`.
