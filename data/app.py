import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

@st.cache_data
def load_data():
    steam1 = pd.read_csv('steam1.csv')
    steam2 = pd.read_csv('steam2.csv')
    steam3 = pd.read_csv('steam3.csv', low_memory=False)
    steam4 = pd.read_csv('steam4.csv')
    steam5 = pd.read_csv('steam5.csv')
    s1 = pd.read_csv('s1.csv')
    s2 = pd.read_csv('s2.csv')
    return steam1, steam2, steam3, steam4, steam5, s1, s2

steam1, steam2, steam3, steam4, steam5, s1, s2 = load_data()

# Sidebar filters
st.sidebar.title("Filter Options")

# Assuming 'tags' column contains game categories
all_genres = s1['tags'].dropna().unique()
selected_genres = st.sidebar.multiselect('Select Game Categories', all_genres, default=all_genres[:5])

# Release year filter
min_year = int(s1['release_year'].min())
max_year = int(s1['release_year'].max())
release_year_range = st.sidebar.slider('Select Release Year Range', min_year, max_year, (min_year, max_year))

# Filter by selected genres
s1_filtered = s1[s1['tags'].isin(selected_genres)]

# Filter by release year
s1_filtered = s1_filtered[(s1_filtered['release_year'] >= release_year_range[0]) & (s1_filtered['release_year'] <= release_year_range[1])]

sentiment = SentimentIntensityAnalyzer()

@st.cache_data
def analyze_sentiment(text):
    if pd.isna(text):
        return np.nan
    scores = sentiment.polarity_scores(text)
    return scores['compound']

if 'sentiment_score' not in s1_filtered.columns:
    s1_filtered['sentiment_score'] = s1_filtered['description'].apply(analyze_sentiment)
st.header('Distribution of Sentiment Scores')

fig_sentiment = px.histogram(
    s1_filtered, 
    x='sentiment_score', 
    nbins=20,
    title='Distribution of Sentiment Scores',
    labels={'sentiment_score': 'Sentiment Score'}
)
st.plotly_chart(fig_sentiment)

st.header('Distribution of Game Prices')

fig_price = px.histogram(
    s1_filtered, 
    x='price_x', 
    nbins=50, 
    title='Distribution of Game Prices',
    labels={'price_x': 'Price ($)'}
)
st.plotly_chart(fig_price)

st.header('Top 10 Game Genres by Review Count')

top_tags = s1_filtered['tags'].value_counts().nlargest(10)
fig_genres = px.bar(
    top_tags, 
    x=top_tags.index, 
    y=top_tags.values,
    title="Top 10 Game Genres by Review Count",
    labels={'x': 'Genre', 'y': 'Review Count'}
)
st.plotly_chart(fig_genres)

st.header('Price vs. Review Score')

fig_price_review = px.scatter(
    s1_filtered, 
    x='price_x', 
    y='reviewscore', 
    hover_name='name', 
    title="Price vs Review Score",
    labels={'price_x': 'Price (USD)', 'reviewscore': 'Review Score'}
)
st.plotly_chart(fig_price_review)

st.header('Average Playtime vs. Review Score')

# Ensure data types
s1_filtered['avgplaytime'] = pd.to_numeric(s1_filtered['avgplaytime'], errors='coerce')
s1_filtered['reviewscore'] = pd.to_numeric(s1_filtered['reviewscore'], errors='coerce')
s1_filtered['copiessold'] = pd.to_numeric(s1_filtered['copiessold'], errors='coerce')

# Drop missing values
s1_clean = s1_filtered.dropna(subset=['avgplaytime', 'reviewscore', 'copiessold'])
s1_clean = s1_clean[s1_clean['copiessold'] > 0]

fig_playtime_review = px.scatter(
    s1_clean,
    x='avgplaytime',
    y='reviewscore',
    size='copiessold',
    title='Average Playtime vs. Review Score',
    labels={'avgplaytime': 'Average Playtime (hrs)', 'reviewscore': 'Review Score'},
    size_max=60,
    opacity=0.7
)
st.plotly_chart(fig_playtime_review)

st.header('Word Cloud of Game Descriptions')

# Generate word cloud
text = ' '.join(s1_filtered['description'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
fig_wc, ax = plt.subplots(figsize=(15, 7.5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig_wc)

st.header('Summary of Data Analysis')

st.markdown("""
- **Total Games Analyzed**: {}
- **Average Sentiment Score**: {:.2f}
- **Top Genre**: {}
- **Average Review Score**: {:.2f}
""".format(
    s1_filtered['name'].nunique(),
    s1_filtered['sentiment_score'].mean(),
    s1_filtered['tags'].value_counts().idxmax(),
    s1_filtered['reviewscore'].mean()
))

st.header('Summary of Data Analysis')

st.markdown("""
- **Total Games Analyzed**: {}
- **Average Sentiment Score**: {:.2f}
- **Top Genre**: {}
- **Average Review Score**: {:.2f}
""".format(
    s1_filtered['name'].nunique(),
    s1_filtered['sentiment_score'].mean(),
    s1_filtered['tags'].value_counts().idxmax(),
    s1_filtered['reviewscore'].mean()
))

st.set_page_config(
    page_title="Steam Game Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Steam Game Analysis Dashboard</p>', unsafe_allow_html=True)
