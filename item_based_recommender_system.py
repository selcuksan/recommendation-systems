###########################################
# Item-Based Collaborative Filtering
###########################################

import pandas as pd


# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################

def create_user_movie_df():
    pd.set_option('display.max_columns', 500)
    ######################################
    # Adım 1: Veri Setinin Hazırlanması
    ######################################
    df_movie = pd.read_csv(
        "Tavsiye Sistemleri/recommender_systems/recommender_systems/datasets/movie_lens_dataset/movie.csv")
    df_rating = pd.read_csv(
        "Tavsiye Sistemleri/recommender_systems/recommender_systems/datasets/movie_lens_dataset/rating.csv")
    df_ = df_movie.merge(df_rating, how="left", on="movieId")
    df = df_.copy()
    ##########################################
    # Adım 2: User Movie Df'inin Oluşturulması
    ##########################################
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    # user_movie_df = common_movies.groupby(["userId", "title"])["rating"].mean().unstack()
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


def item_based_recommender(movie_name, count=5):
    ######################################
    # Adım 3: Item-Based Film Önerilerinin Yapılması
    ######################################
    try:
        movie_name = user_movie_df[movie_name]
    except Exception:
        return "Film ismi yanlış"
    recommendation = user_movie_df.corrwith(movie_name).sort_values(ascending=False)[:count]

    return recommendation


def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]


user_movie_df = create_user_movie_df()

check_film("Titanic", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
item_based_recommender(movie_name)
