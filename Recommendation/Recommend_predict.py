from keybert import KeyBERT
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import json
from tqdm import tqdm
import logging


import nltk
nltk.download("popular")

movies = pd.read_csv('data/Movie_data.csv')
song = pd.read_csv('data/Song_data.csv')

movie_w2v = KeyedVectors.load_word2vec_format("model/movie_w2v")
song_w2v = KeyedVectors.load_word2vec_format("model/song_w2v")

kw_model = KeyBERT()

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename = "logfile.log",
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.INFO)

logger = logging.getLogger()

def preprocess_text(text):

    key1 = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words="english",top_n=5)
    key2 = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 2), stop_words="english",use_mmr=True, diversity=0.7)
    key3 = kw_model.extract_keywords(text, keyphrase_ngram_range=(3, 3), stop_words="english",use_maxsum=True, nr_candidates=20)

    key_list = []
    for k1 in key1:
        key_list.append(k1[0])

    for k2 in key2:
        key_list.append(k2[0])

    for k3 in key3:
        key_list.append(k3[0])

    preprocessed_text = ' '.join(key_list)

    return preprocessed_text

def get_vector(text, model):
    vectors = []
    for word in word_tokenize(text):
        if word in model.key_to_index:
            vectors.append(model[word])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def recommend_movie(text, top_n=5):
    # 입력된 텍스트를 전처리합니다.
    preprocessed_text = preprocess_text(text)

    # 입력된 텍스트의 벡터를 구합니다.
    text_vector = get_vector(preprocessed_text, movie_w2v)

    # 각 영화 데이터의 벡터를 구합니다.
    movie_vectors = []
    movie_title = []
    movie_director = []

    for i,overview in enumerate(movies['Description']):
        try:
            movie_vectors.append(get_vector(preprocess_text(overview), movie_w2v))
            movie_title.append(movies['Title'][i])
            movie_director.append(movies['Director'][i])
        except:
            pass

    # 입력된 텍스트와 각 영화 데이터의 유사도를 측정합니다.
    similarities = []
    for i,movie_vector in tqdm(enumerate(movie_vectors)):
        similarities.append(cosine_similarity(text_vector.reshape(1, -1), movie_vector.reshape(1, -1)).flatten()[0])
        logger.error(f"Tqdm{i}")

    # 유사도가 가장 높은 상위 n개의 영화를 추천합니다.
    top_n_movies = pd.DataFrame({'Title': movie_title, 'Director': movie_director ,'similarity': similarities}) \
        .sort_values(by='similarity', ascending=False).head(top_n).reset_index(drop=True)
    
    return top_n_movies

def recommend_song(text, top_n=5):
    # 입력된 텍스트를 전처리합니다.
    preprocessed_text = preprocess_text(text)

    # 입력된 텍스트의 벡터를 구합니다.
    text_vector = get_vector(preprocessed_text, song_w2v)

    # 각 영화 데이터의 벡터를 구합니다.
    song_vectors = []
    song_title = []
    song_artist = []

    for i,Lyric in enumerate(song['Lyric']):
        try:
            song_vectors.append(get_vector(preprocess_text(Lyric),song_w2v))
            song_title.append(song['Title'][i])
            song_artist.append(song['Artist'][i])
        except:
            pass

    # 입력된 텍스트와 각 영화 데이터의 유사도를 측정합니다.
    similarities = []
    for i,song_vector in tqdm(enumerate(song_vectors)):
        similarities.append(cosine_similarity(text_vector.reshape(1, -1), song_vector.reshape(1, -1)).flatten()[0])
        logger.error(f"Tqdm_song{i}")

    # 유사도가 가장 높은 상위 n개의 영화를 추천합니다.
    top_n_song = pd.DataFrame({'Title': song_title, 'Artist': song_artist ,'similarity': similarities}) \
        .sort_values(by='similarity', ascending=False).head(top_n).reset_index(drop=True)

    return top_n_song

def text_generator(file_root):
    text = []
    f = open(file_root, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip() # 줄 끝의 줄 바꿈 문자를 제거한다.
        text.append(line)
    f.close()
    str = ''.join(text)
    return str

def write_json(movie_df,song_df):
    recommend_movie = []
    recommend_song = []

    for i,_ in enumerate(movie_df):
        recommend_movie.append({'title':movie_df['Title'][i],'producer':movie_df['Director'][i]})
    
    for j,_ in enumerate(song_df):
        recommend_song.append({'title':song_df['Title'][j],'producer':song_df['Artist'][j]})
    
    recomend = {'movie':recommend_movie,'song':recommend_song}
    return recomend

def recommend(args):
    file_root = f'test/{args.user_id}/diary/{args.diary_id}/content.txt'
    text = text_generator(file_root)
    recommend_movie_df = recommend_movie(text)
    recommend_song_df = recommend_song(text)
    recommend = write_json(recommend_movie_df,recommend_song_df)
    file_path = f"test/{args.user_id}/recommend.json"
    with open(file_path, 'w') as outfile:
        json.dump(recommend, outfile, indent=4)

def get_args():
    parser = argparse.ArgumentParser(
        description='Tell me which file I should do the recommended analysis')
    parser.add_argument('--user_id', required=True, type=str,
                        help='Please enter analysis text')
    parser.add_argument('--diary_id', required=True, type=str,
                        help='Please enter analysis text')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    recommend(args)



