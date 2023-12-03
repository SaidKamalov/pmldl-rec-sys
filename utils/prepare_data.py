import os
import sys

PROJECT_PATH = os.sep + os.path.join(*__file__.split(os.sep)[:-2])
os.chdir(PROJECT_PATH)
sys.path.append(PROJECT_PATH)


import pandas as pd
import re
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


def get_movie_table(df: pd.DataFrame, clean_titles=False):
    """Preprocess movie table.

    Params:
        df (pd.DataFrame): movie Dataframe
        clean_titles (bool): flag to preprocess movie titles or not

    Returns:
        df (pd.DataFrame): preprocessed Dataframe
    """
    df = df.drop(["video_release_date", "url"], axis=1)
    df = df.dropna(axis=0)

    # clean titles
    if clean_titles:
        df["title"] = df["title"].apply(clean_titles)

    # replace date with year only
    df["year"] = df["release_date"].apply(lambda x: int(str(x).split(sep="-")[-1]))
    df = df.drop(["release_date"], axis=1)
    return df


def clean_titles(title: str):
    words = re.sub(r"[^\w\s]", "", title.strip().lower()).split()
    clean_title = " ".join([w for w in words if w.isalnum() and not w.isdigit()])
    return clean_title


def get_users_table(df: pd.DataFrame, columns_to_encode: list[str]):
    """Preprocess movie table.

    Params:
        df (pd.DataFrame): users Dataframe
        columns_to_encode (list[str]): names of columns for OHE

    Returns:
        df (pd.DataFrame): preprocessed Dataframe
    """
    df = df.drop(["zip_code"], axis=1)
    df = df.dropna(axis=0)

    # encode categorical columns
    dummies = []
    for column_name in columns_to_encode:
        encoded = pd.get_dummies(df[column_name]).astype(int)
        dummies.append(encoded)
    df = pd.concat([df] + dummies, axis=1)
    df = df.drop(columns_to_encode, axis=1)
    return df


def process_rating_table(path):
    """Get ratings Dataframe.

    Params:
        path (str): path to file with ratings table

    Returns:
        ratings (pd.DataFrame): preprocessed ratings Dataframe
    """
    ratings = pd.read_csv(path, sep="\t", header=None)
    ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = ratings.drop(["timestamp"], axis=1)
    return ratings


"""Functions to preprocess data to work with surprise library:"""


def create_custom_surprise_dataset(df: pd.DataFrame, rating_scale=(1, 5)):
    reader = Reader(rating_scale=rating_scale)  # Define the rating scale if necessary
    custom_dataset = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)
    return custom_dataset


def split_train_test(dataset, test_size, random_state, shuffle=True):
    train, test = train_test_split(
        dataset, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    return train, test
