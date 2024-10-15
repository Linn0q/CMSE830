import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Steam Game Data Analysis", layout="wide")
# 1. 项目介绍和基础数据描述
st.title('My project-Steam Game Data Analysis')
st.write('EDA')
# Function to load data
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

sentiment = SentimentIntensityAnalyzer()

# Analyze sentiment of the 'Description' column
def analyze_sentiment(text):
    if pd.isna(text):
        return np.nan
    scores = sentiment.polarity_scores(text)
    return scores['compound']  # Using the compound score as overall sentiment

st.sidebar.header('User Input Features')

# Sidebar options for data selection
if st.sidebar.button('Analyze Sentiment Scores'):
    st.subheader('Sentiment Analysis of Game Descriptions')
    with st.spinner('Analyzing sentiment...'):
        steam1['sentiment_score'] = steam1['description'].apply(analyze_sentiment)
        st.write(steam1['sentiment_score'].describe())
else:
    if 'sentiment_score' not in steam1.columns:
        steam1['sentiment_score'] = steam1['description'].apply(analyze_sentiment)
        
# 2. 分析游戏价格的直方图：游戏市场的价格分布概览
st.subheader('Distribution of Game Prices')
nbins = st.slider('Select number of bins for histogram', min_value=10, max_value=100, value=50, step=10)
fig1 = px.histogram(s1, x='price_x', nbins=nbins, title='Distribution of Game Prices')
fig1.update_xaxes(title='Price ($)')
fig1.update_yaxes(title='Number of Games')
st.plotly_chart(fig1)

# 3. 游戏类型与评价：展示用户对不同类型游戏的评价数
st.subheader('Top Game Genres by Review Count')
num_genres = st.slider('Select number of top genres to display', min_value=5, max_value=20, value=10, step=1)
top_tags = s1['tags'].value_counts().nlargest(num_genres)
fig2 = px.bar(top_tags, x=top_tags.index, y=top_tags.values,
              title=f"Top {num_genres} Game Genres by Review Count",
              labels={'x': 'Genre', 'y': 'Review Count'})
st.plotly_chart(fig2)

# 4. 情感得分分布：通过分析游戏描述进行情感得分
st.subheader('Distribution of Sentiment Scores')
nbins_sentiment = st.slider('Select number of bins for sentiment score histogram', min_value=10, max_value=50, value=20, step=5)
fig3 = px.histogram(steam1, x='sentiment_score', nbins=nbins_sentiment,
                    title='Distribution of Sentiment Scores',
                    labels={'sentiment_score': 'Sentiment Score'})
st.plotly_chart(fig3)

# 5. 游戏价格与评分：展示价格和评分的关系，揭示用户行为与价格策略的影响
st.subheader('Scatter Plot of Price vs Review Score')
x_var = st.selectbox('Select x-axis variable', options=['price_x', 'avgplaytime', 'copiessold', 'revenue'], index=0)
y_var = st.selectbox('Select y-axis variable', options=['reviewscore', 'avgplaytime', 'copiessold', 'revenue'], index=1)
fig4 = px.scatter(s1, x=x_var, y=y_var, hover_name='name', title=f"{x_var} vs {y_var}",
                  labels={x_var: x_var, y_var: y_var})
st.plotly_chart(fig4)

# 6. 最高评价游戏：分析评论最多的游戏，展示用户对游戏的喜好度
st.subheader('Top Games by Number of Reviews')
top_n = st.slider('Select number of top games to display', min_value=10, max_value=50, value=20, step=10)
top_reviews = s1.nlargest(top_n, 'review_no')
fig5 = px.bar(top_reviews, x='name', y='review_no', title=f'Top {top_n} Games by Number of Reviews')
fig5.update_xaxes(title='Game', tickangle=45)
fig5.update_yaxes(title='Number of Reviews')
st.plotly_chart(fig5)

# 7. 用户反馈情感的分布图 (KDE 图)
st.subheader('KDE Plot of Sentiment Scores')
plt.figure(figsize=(10, 6))
sns.kdeplot(merged_data['sentiment_score'].dropna(), shade=True)
plt.title('KDE Plot of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Density')
st.pyplot(plt.gcf())

# 8. 热图：数值变量的相关性分析，展示特征之间的关系
st.subheader('Correlation Heatmap of Numerical Variables')
numeric_cols = ['review_no', 'copiessold', 'revenue', 'avgplaytime', 'reviewscore']
corr_matrix = s1[numeric_cols].corr()
fig10 = px.imshow(corr_matrix, text_auto=True, title='Correlation Heatmap of Numerical Variables')
st.plotly_chart(fig10)

# 9. 词云：展示游戏评论中最常见的词汇，揭示用户关注的内容
st.subheader('Word Cloud of Game Reviews')
text_reviews = ' '.join(steam3['review'].dropna()) 
wordcloud_reviews = WordCloud(width=800, height=400, background_color='white').generate(text_reviews) 
st.image(wordcloud_reviews.to_array(), use_column_width=True)

# 10. 开发者分布的地图可视化
st.subheader('Developer Locations')
developer_locations = pd.DataFrame({
    'Developer': ['Dev A', 'Dev B', 'Dev C'],
    'Latitude': [37.7749, 51.5074, 35.6895],
    'Longitude': [-122.4194, -0.1278, 139.6917],
})
fig11 = px.scatter_mapbox(developer_locations, lat='Latitude', lon='Longitude', hover_name='Developer',
                          zoom=1, height=500)
fig11.update_layout(mapbox_style="open-street-map")
fig11.update_layout(title='Developer Locations')
st.plotly_chart(fig11)
