#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dosyanın okunması
df = pd.read_csv(
    "Tavsiye Sistemleri/recommender_systems/recommender_systems/datasets/the_movies_dataset/movies_metadata.csv",
    low_memory=False)  # DtypeWarning kapamak icin


#################################
# 2. Cosine similaritiy Matrisinin Oluşturulması
#################################


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


#################################
# 3. Benzerliklere göre önerilerin Oluşturulması
#################################

def content_based_recommender(title, cosine_sim_matrix, dataframe, count=10):
    # indexleri olusturma
    indices = pd.Series(dataframe.index, index=dataframe["title"])
    indices = indices[~indices.index.duplicated(keep="last")]
    # title'ın indexini yakalama
    movie_index = indices[title]
    # title'a benzerlik scorelarını hesaplama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # film tavsiyelerini getirme
    movie_indices = similarity_scores.sort_values(by="score", ascending=False)[:count].index
    return df["title"].loc[movie_indices]


cosine_sim = calculate_cosine_sim(df)

content_based_recommender("Honeymoon", cosine_sim, df, count=10)
content_based_recommender("From Mexico With Love", cosine_sim, df, count=10)
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
