from models.svd_model import get_recommendations, get_model

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Get user ID from command line")
    parser.add_argument("--user_id", type=int, help="The user ID", required=True)
    parser.add_argument(
        "--num_of_reqs",
        type=int,
        help="Number of recommendations to get",
        default=5,
        required=False,
    )
    parser.add_argument(
        "--path_to_model",
        type=str,
        help="path to model checkpoint",
        default="models/svd_saved.dump",
        required=False,
    )

    args = parser.parse_args()
    user_id = args.user_id
    path_to_model = args.path_to_model
    n = args.num_of_reqs

    geners = pd.read_csv("data/raw/u.genre", sep="|", header=None)
    geners.columns = ["name", "id"]
    geners_list = list(geners["name"])

    movies = pd.read_csv("data/raw/u.item", sep="|", encoding="latin-1", header=None)
    movies.columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "url",
    ] + geners_list
    model = get_model(path_to_model)

    results = get_recommendations(
        user_id=user_id, movies_table=movies, model=model, n=n
    )
    for i, res in enumerate(results):
        title = res[0]
        id = res[1]
        estimated_rating = res[2]
        print(f"{i+1}. {title}, est. rating = {estimated_rating}")


if __name__ == "__main__":
    main()
