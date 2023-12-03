import os
import sys

PROJECT_PATH = os.sep + os.path.join(*__file__.split(os.sep)[:-2])
os.chdir(PROJECT_PATH)
sys.path.append(PROJECT_PATH)


from surprise import SVD, dump, accuracy
from surprise.model_selection import GridSearchCV
from collections import defaultdict

from utils.prepare_data import process_rating_table


def get_recommendations(user_id, movies_table, model, n=5):
    """Get movie recommendations for specific user

    Params:
        user_id (int): user id
        movies_table (pd.Dataframe): dataframe with movies info
        model: fitted model
        n (int): number of recommendations

    Returns:
    tuple of title, movie_id, estimated rating
    """
    # user_id film_id rating
    ratings = process_rating_table("data/raw/u.data")

    # Get the list of all item IDs
    all_item_ids = ratings["movie_id"].unique()

    # Remove the item IDs that the user has already rated
    rated_item_ids = ratings[ratings["user_id"] == user_id]["movie_id"].values
    unrated_item_ids = [
        item_id for item_id in all_item_ids if item_id not in rated_item_ids
    ]

    # get predictions
    predictions = [model.predict(uid=user_id, iid=id) for id in unrated_item_ids]
    predictions_sorted = sorted(predictions, key=lambda x: x.est, reverse=True)

    results = []

    for i in range(n):
        raw = movies_table[movies_table["movie_id"] == predictions_sorted[i].iid]
        title = raw["title"].values[0]
        results.append((title, predictions_sorted[i].iid, predictions_sorted[i].est))

    return results


def get_model(path_to_model="models/svd_saved.dump"):
    """Load the model from memory"""
    assert os.path.exists(path_to_model), "please, train and save the model before"
    _, model = dump.load(path_to_model)
    return model


def train(train_dataset, dataset=None, use_grid_search=False, path_to_save=None):
    """Train the model and save it, if path is specified

    Params:
        train_dataset: dataset to train on
        dataset: initial unsplitted dataset (only needed for grid_search)
        use_grid_search (bool): use grid search or not
        path_to_save (str): path to save the model

    Returns:
        model: fitted model
    """
    model = SVD()
    if use_grid_search:
        print("start grid search")
        param_grid = {
            "n_factors": [25, 50, 100],  # The number of factors
            "n_epochs": [20, 30, 40],  # The number of iteration of the SGD procedure
            "lr_all": [0.002, 0.005, 0.01],  # The learning rate for all parameters
            "reg_all": [0.02, 0.05, 0.1],  # The regularization term for all parameters
        }
        gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=5, n_jobs=4)
        gs.fit(dataset)
        model = gs.best_estimator["rmse"]
    model.fit(train_dataset)

    if path_to_save:
        dump.dump(path_to_save, algo=model)

    return model


def eval(model, test_dataset, k=5, threshold=3.5):
    """Evaluate the model.

    Params:
        model (surprise.prediction_algorithms): model itself
        test_dataset: data for evaluation
        k (int): number of top predicted items
        threshold (float): threshold to consider if item is recommended or not and also to consider relevant items

    Returns:
        rmse (float): Root Mean Squared Error
        avg_precision_at_k (float): average precision at k metrics
        avg_recall_at_k (float): average recall at k metrics
    """
    predictions = model.test(test_dataset)
    rmse = accuracy.rmse(predictions)
    avg_precision_at_k, avg_recall_at_k = precision_recall_at_k(
        predictions, k, threshold
    )
    return rmse, avg_precision_at_k, avg_recall_at_k


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return average precision and recall at k metrics.

    Params:
        k (int): number of top predicted items
        threshold (float): threshold to consider if item is recommended or not and also to consider relevant items

     Returns:
        avg_precision_at_k (float): average precision at k metrics
        avg_recall_at_k (float): average recall at k metrics

    Reference:https://surprise.readthedocs.io/en/stable/FAQ.html?highlight=precision#how-to-compute-precision-k-and-recall-k
    """

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        precisions.append(precision)

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        recalls.append(recall)

    avg_precision = sum(precisions) / len(user_est_true.items())
    avg_recall = sum(recalls) / len(user_est_true.items())

    return avg_precision, avg_recall
