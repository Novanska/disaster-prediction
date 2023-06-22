import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('popular')
def run():
    #load model
    best_model = load_model('best_model')
    #pembuatan title
    st.title('Disaster Tweet Prediction')
    st.subheader('Predict disaster or not')
    st.markdown('---')
    with st.form(key= 'form_tweet'):
        st.markdown('### **Tweet**')
        tweets_text = st.text_input('',value= '')
        submitted = st.form_submit_button('Predict')

    #pembuatan stopwords 
    new_stopwords = ['t','rt','co']
    stopwords_eng = stopwords.words("english")
    stopwords_eng = stopwords_eng + new_stopwords
    stopwords_eng = list(set(stopwords_eng))

    #pembuatan pemrosesan tweets
    lemmatizer = WordNetLemmatizer()
    def proses_tweets(tweets):
        tweets = tweets.lower() # Mengubah Teks ke Lowercase
        tweets = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', tweets) # Menghilangkan http
        tweets = re.sub("@[A-Za-z0-9_]+", " ", tweets)  # Menghilangkan mention
        tweets = re.sub("#[A-Za-z0-9_]+", " ", tweets) # Menghilangkan hashtag
        tweets = re.sub(r"\\n", " ",tweets) # Menghilangkan \n (enter)
        tweets = tweets.strip() # Menghilangkan whitespace
        tweets = re.sub("[^A-Za-z\s']", " ", tweets) # Menghilangkan emoji atau atau simbol matematika
        tweets = re.sub("\s\s+" , " ", tweets) # Menghilangkan double spasi
        tokens = word_tokenize(tweets) # Melakukan tokenize
        tweets = ' '.join([word for word in tokens if word not in stopwords_eng]) # Menghilangkan stopwords
        tweets = lemmatizer.lemmatize(tweets) # Melakukan lemmatize
        return tweets

    data_inf = {
        'text' : tweets_text                               
        }
    data_inf = pd.DataFrame([data_inf])
    if submitted:
        data_inf['tweet_preprocessing'] = data_inf['text'].apply(lambda x : proses_tweets(x))
        #predict model  
        y_inf_pred = np.argmax(best_model.predict(data_inf['tweet_preprocessing']), axis=-1)

        if y_inf_pred[0] == 0:
            hasil = 'non-disaster'
        else:
            hasil = 'disaster'
        
        st.write('This tweet is : ',hasil)
        
if __name__ == '__main__':
    run()