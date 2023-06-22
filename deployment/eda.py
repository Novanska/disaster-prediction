import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title = 'DISASTER PREDICTION' ,
    initial_sidebar_state= 'expanded',
)

def run():
    # membuat title
    st.title('CHURN PREDICTION')
    st.subheader('EXPLORATORY DATA ANALYSIS(EDA)')
    st.markdown('---')

    # menambahkan gambar
    image = Image.open('disaster.jpg')
    st.image(image)
    st.write('## Background')
    st.write('''There's many people can tweet freely in the twitter, but we didn`t know wheter is disaster or not''')
    st.write('## Objective')
    st.write('''predict the tweet is disaster or not.''')
            
    st.write('Dataset : ')
    data = pd.read_csv('set-train.csv')
    st.dataframe(data)
    st.write('Disaster and Non Disaster Distribution')
    fig = plt.figure(figsize=(8,8))
    plt.pie(data['target'].value_counts(),labels = [1,0], autopct='%.0f%%', explode=[0,0.1])
    plt.title('Disaster VS Non Disaster')
    st.pyplot(fig)
    st.write('Top 10 Keywords')
    fig = plt.figure(figsize=(15,5))
    keywords = pd.DataFrame({'Count': data['keyword'].value_counts()})
    sns.barplot(y=keywords[0:10].index, x = keywords[0:10]['Count'])
    plt.title('Top 10 keywords')
    st.pyplot(fig)
    st.write('Top 10 keywords disaster or non disaster')
    fig = plt.figure(figsize=(15,5))
    disaster_word = data.loc[data['target']==1]['keyword'].value_counts()
    nondisaster_word = data.loc[data['target']==0]['keyword'].value_counts()
    fig,ax= plt.subplots(1,2, figsize=(15,5))
    sns.barplot(y=disaster_word[0:10].index, x=disaster_word[0:10], ax=ax[0])
    sns.barplot(y=nondisaster_word[0:10].index, x=nondisaster_word[0:10],ax=ax[1])
    ax[0].set_title("Top 10 Disaster Tweets")
    ax[0].set_xlabel("Keyword Frequency")
    ax[1].set_title("Top 10 Non-Disaster Tweets")
    ax[1].set_xlabel("Keyword Frequency")
    plt.tight_layout()
    st.pyplot(fig)
    # top 10 locations 
    fig = plt.figure(figsize=(15,5))
    st.write('Top 10 Locations')
    locations_vc = data["location"].value_counts()
    sns.barplot(y=locations_vc[0:10].index, x=locations_vc[0:10])
    plt.title("Top 10 Locations")
    st.pyplot(fig)
    fig = plt.figure(figsize=(15,5))
    st.write('Top 10 Locations Disaster and Non Disaster Tweets')
    #check top 10 locations
    disaster_locations = data.loc[data["target"] == 1]["location"].value_counts()
    nondisaster_locations = data.loc[data["target"] == 0]["location"].value_counts()

    fig, ax = plt.subplots(1,2, figsize=(20,8))
    sns.barplot(y=disaster_locations[0:10].index, x=disaster_locations[0:10],ax=ax[0])
    sns.barplot(y=nondisaster_locations[0:10].index, x=nondisaster_locations[0:10],ax=ax[1])
    ax[0].set_title("Top 10 Locations - Disaster Tweets")
    ax[0].set_xlabel("Keyword Frequency")
    ax[1].set_title("Top 10 Locations - Non-Disaster Tweets")
    ax[1].set_xlabel("Keyword Frequency")
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == '__main__':
    run()