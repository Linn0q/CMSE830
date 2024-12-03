import streamlit as st

page = st.sidebar.selectbox("Select Page🛗", ["Welcome🎮","Data🌈", "EDA🌷","WordCloud🌨️","Models🏈","Recommendation🎯","Conclusion🍩",])

if page == "Welcome🎮":
    st.title("Welcome🎮")
    st.image("https://wallpapers.com/images/high/steam-platform-logo-wallpaper-xf5abpdtbsakirq5.webp", use_column_width=True)
    st.markdown("# Hello!🌈")
    st.markdown("### Welcome to my CMSE-830 project!")
    st.write("Do you like games? Do you want to explore more about games?")
    st.write("As a game lover, a data science student and someone who worked in game industy, I am always interested in the data behind games. ")
    st.write("Today we have a lot of opportunities to look for games'rating and give comments.")
    st.write("But what truly influences these ratings? Is it the genre, the gameplay, or perhaps the emotional connection a player has with the game?")
    st.write("Let's find out with me!")
    st.write("In this project, I aim to dive deep into the data behind games on Steam. I will explore how various factors such as game genre, publisher, and so on correlate with overall game ratings.")
    st.write("My goal is to discover insights that can help both gamer players and developers understand what drives game popularity, and how emotional engagement reflected in user comments relates to game success.")
    st.write("And after analysis I hope that I can answer questions that every gamer players or developer wonders about: ")
    st.markdown('*How do player sentiments influence game ratings?*')
    st.markdown('*What makes a great game?*')
    st.write("")
    st.markdown('- **Page Data🌈** displays all of datasets and the detailed cleaning processing is on my Github.')
    st.markdown('- **Page EDA🌷** displays my EDA (Exploratory Data Analysis) work. Through a series of visualizations, we can find patterns, trends, and insights hidden within the data.')
    st.markdown('- **Page WordCloud🌨️** displays my wordcloud analysis. We can discover the hidden trends and patterns in the words.')
    st.markdown('- **Page Model🏈** displays my analysis of recommendation system models, comparing their accuracy and performance to identify the best model for personalized recommendations.')
    st.markdown('- **Page Recommendation🎯** displays my game recommendation system, helping users discover personalized game suggestions based on their preferences.')
    st.markdown('- **Page Conclusion🍩** summarizes the results of my analysis and we can discuss together!')
    st.markdown('**Join me on this journey and to understand better the art and science behind games🎉**')
    st.markdown('**Enjoy!🍕🍔🍟🍨🍪🍫🍬🍰**')
  
elif page == "Data🌈":

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.experimental import enable_iterative_imputer  # Enable if using older scikit-learn
    from sklearn.impute import IterativeImputer

    steam1 = pd.read_csv('steam1.csv')
    steam2 = pd.read_csv('steam2.csv')
    steam3 = pd.read_csv('s3.csv')
    steam4 = pd.read_csv('steam4.csv')
    steam5 = pd.read_csv('steam5.csv')
    st.title("Data🌈")
 
    st.markdown("### 🔗 Data Sources")
    st.markdown("The datasets used in this project were sourced from the following Kaggle repositories:")
    st.markdown("- [Steam Games Reviews and Rankings](https://www.kaggle.com/datasets/mohamedtarek01234/steam-games-reviews-and-rankings)")
    st.markdown("- [Top 1500 Games on Steam by Revenue (09-09-2024)](https://www.kaggle.com/datasets/alicemtopcu/top-1500-games-on-steam-by-revenue-09-09-2024)")
    st.markdown("- [Steam Games Dataset](https://www.kaggle.com/datasets/wajihulhassan369/steam-games-dataset)")

    st.markdown("These datasets provide a comprehensive overview of Steam games, including reviews, rankings, sales data, and user feedback. They form the foundation for the analysis and recommendation system development in this project. 🚀")

    st.markdown("### 📊 Dataset Overview")
    st.markdown("In this project, I utilized five datasets from **Kaggle**, which provide diverse information about Steam games. Here’s a closer look at each dataset:")

    st.markdown("- **Dataset 1 (steam1.csv) 🎮**:")
    st.markdown("  - **Description**: This dataset provides a rich overview of games available on Steam, focusing on their descriptions, user reviews, and popularity.")
    st.markdown("  - **Key Columns**:")
    st.markdown("    - `name`: The name of the game.")
    st.markdown("    - `description`: A detailed textual description of the game.")
    st.markdown("    - `review_no`: The number of user reviews for each game.")
    st.markdown("    - `reviewscore`: The overall score of the game, based on user reviews.")
    st.markdown("    - `tags`: Tags associated with the game, such as genre or gameplay features.")
    st.markdown("  - **Purpose**: This dataset is ideal for analyzing the relationship between game descriptions, tags, and their popularity, as well as exploring user sentiment and game features.")
    st.markdown("  - **Source**: [Steam Games Reviews and Rankings](https://www.kaggle.com/datasets/mohamedtarek01234/steam-games-reviews-and-rankings)")
    st.dataframe(steam1.head())

    st.markdown("- **Dataset 2 (steam2.csv) 🛒**:")
    st.markdown("  - **Description**: This dataset focuses on the financial aspects of Steam games, including their pricing, sales performance, and review scores.")
    st.markdown("  - **Key Columns**:")
    st.markdown("    - `price`: The price of the game (in USD).")
    st.markdown("    - `revenue`: The total revenue generated by the game.")
    st.markdown("    - `copiessold`: The number of copies sold for each game.")
    st.markdown("    - `reviewscore`: The overall user rating for the game.")
    st.markdown("  - **Purpose**: This dataset is instrumental for analyzing how game pricing strategies and sales volumes affect user satisfaction and revenue generation. It also provides a basis for exploring relationships between price, popularity, and reviews.")
    st.dataframe(steam2.head())

    st.markdown("- **Dataset 3 (steam3.csv) ✨**:")
    st.markdown("  - **Description**: This dataset provides advanced metrics derived from user reviews, including sentiment scores and weighted average ratings for games.")
    st.markdown("  - **Key Columns**:")
    st.markdown("    - `game_name`: The name of the game.")
    st.markdown("    - `sentiment_score`: A numerical value representing the average sentiment derived from user reviews.")
    st.markdown("    - `weighted_rating`: A weighted average score based on review sentiments and user ratings.")
    st.markdown("    - `review_count`: The total number of user reviews for each game.")
    st.markdown("  - **Purpose**: This dataset is essential for understanding how user sentiments and ratings collectively influence a game's perceived quality. It enables the correlation of emotional feedback with quantitative scores.")
    st.markdown("  - **Source**: [Steam Games Reviews and Rankings](https://www.kaggle.com/datasets/mohamedtarek01234/steam-games-reviews-and-rankings)")
    st.dataframe(steam3.head())

    st.markdown("- **Dataset 4 (steam4.csv) 🏷️**:")
    st.markdown("- **Dataset 4 (steam4.csv) 🏷️**:")
    st.markdown("  - **Description**: This dataset provides information about game genres, tags, and rankings on Steam, enabling an in-depth analysis of user preferences across different game categories.")
    st.markdown("  - **Key Columns**:")
    st.markdown("    - `name`: The name of the game.")
    st.markdown("    - `genre`: The genre of the game (e.g., Action, Adventure, RPG).")
    st.markdown("    - `tags`: A list of tags describing the game's features and themes.")
    st.markdown("    - `rank`: The ranking of the game based on revenue.")
    st.markdown("    - `review_type`: A categorical label indicating the overall review sentiment (e.g., Positive, Negative).")
    st.markdown("  - **Purpose**: This dataset is particularly valuable for understanding how genre and tags influence a game's revenue and user feedback. It also helps explore the correlation between game rankings and review sentiments.")
    st.markdown("  - **Source**: [Top 1500 Games on Steam by Revenue (09-09-2024)](https://www.kaggle.com/datasets/alicemtopcu/top-1500-games-on-steam-by-revenue-09-09-2024)")
    st.dataframe(steam4.head())

    st.markdown("- **Dataset 5 (steam5.csv) 🌟**:")
    st.markdown("  - **Description**: This dataset focuses on overall player ratings and feedback summaries for Steam games, offering a comprehensive view of player satisfaction and game success.")
    st.markdown("  - **Key Columns**:")
    st.markdown("    - `name`: The name of the game.")
    st.markdown("    - `overall_player_rating`: The aggregated player rating for the game.")
    st.markdown("    - `review_type`: A label categorizing reviews as Positive or Negative.")
    st.markdown("    - `reviewscore`: The user-generated score indicating the overall quality of the game.")
    st.markdown("    - `rating_count`: The total number of player ratings.")
    st.markdown("  - **Purpose**: This dataset is ideal for evaluating the general reception of games among players. It can be used to analyze trends in player feedback, identify factors driving high ratings, and correlate ratings with other performance metrics.")
    st.markdown("  - **Source**: [Steam Games Dataset](https://www.kaggle.com/datasets/wajihulhassan369/steam-games-dataset)")
    st.dataframe(steam5.head())

    st.markdown("### 🧹 Data Cleaning Overview")
    st.markdown("To ensure high-quality analysis and accurate modeling, a thorough data cleaning process was implemented. Here's a summary of the main steps, accompanied by some exciting highlights: ")

    st.markdown("- **1. Handling Missing Values 🔍**: Identified missing entries across datasets and applied tailored strategies, including:")
    st.markdown("  - **MICE (Multiple Imputation by Chained Equations)**: Used advanced imputation methods to estimate missing values in numerical fields (e.g., `revenue`, `reviewscore`). This technique considers correlations between features, ensuring more accurate imputations.")
    st.markdown("  - **Default Values**: Replaced missing categorical fields (e.g., `tags`, `review_type`) with placeholders like 'Unknown'.")
    st.markdown("  - **Dropping Rows**: Removed rows with excessive missing information when necessary.")
    st.image("https://github.com/Linn0q/CMSE830/blob/main/MICE1.png?raw=true", use_column_width=True)
    st.image("https://github.com/Linn0q/CMSE830/blob/main/MICE2.png?raw=true", use_column_width=True)
    st.markdown("- **2. Removing Duplicates 🗑️**: Checked for duplicate entries using game names and unique IDs, ensuring no redundant data skewed the results.")

    st.markdown("- **3. Standardizing Formats 📏**:")
    st.markdown("  - Unified date formats to `YYYY-MM-DD` for consistency.")
    st.markdown("  - Normalized text fields like `tags` and `genres` for easier processing (e.g., lowercasing, trimming spaces).")
    st.markdown("  - Converted numerical data types for efficient computation.")

    st.markdown("- **4. Dealing with Outliers 🚨**: Applied robust statistical techniques to detect and manage outliers:")
    st.markdown("  - Used **IQR (Interquartile Range)** to identify extreme values in fields like `price` and `revenue`.")
    st.markdown("  - Adjusted or removed outliers to maintain data quality.")

    st.markdown("- **5. Data Enrichment and Feature Engineering ✨**:")
    st.markdown("  - Created new features, such as:")
    st.markdown("    - `review_density`: Number of reviews per unit of sales.")
    st.markdown("    - `price_per_hour`: Price divided by average playtime.")
    st.markdown("  - Enhanced datasets by merging sentiment scores and weighted averages.")

    st.markdown("- **6. Data Merging 🔗**: Seamlessly merged datasets using common keys like `name` and `game_id` to consolidate information from various sources.")

    st.markdown("**All of datasets provide the foundation for my exploratory data analysis, modeling, and recommendation system development. Let's dive in and explore the art and science behind games! ✨**")

elif page == "EDA🌷":
    st.title("EDA🌷")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.figure_factory as ff
    from wordcloud import WordCloud
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#   from textblob import TextBlob
    from sklearn.impute import SimpleImputer
    import streamlit as st
    # Load datasets
    steam1 = pd.read_csv('steam1.csv')
    steam2 = pd.read_csv('steam2.csv')
    steam4 = pd.read_csv('steam4.csv')
    steam5 = pd.read_csv('steam5.csv')
    s1 = pd.read_csv('s1.csv')
    s2 = pd.read_csv('s2.csv')


    sentiment = SentimentIntensityAnalyzer()

    def analyze_sentiment(text):
        if pd.isna(text):
            return np.nan
        scores = sentiment.polarity_scores(text)
        return scores['compound']

    steam1['sentiment_score'] = steam1['description'].apply(analyze_sentiment)
    s1['sentiment_score'] = s1['description'].apply(analyze_sentiment)
    merged_data = pd.merge(s1, s2, left_on='name', right_on='game_name', how='outer')
    merged_data['sentiment_score'] = merged_data['description'].apply(analyze_sentiment)
    merged_data['sentiment_score'].describe()

    st.title('Steam Games Analysis Dashboard')

    #1
    st.subheader('Top 10 Games by Review Count')
    top_n = st.slider('Select the number of top games to display', min_value=5, max_value=50, value=10)
    top_games = steam1.nlargest(top_n, 'review_no')
    fig1 = px.bar(top_games, y='name', x='review_no', orientation='h', title="Top 10 Games by Review Count")
    fig1.update_xaxes(title="Number of Reviews")
    fig1.update_yaxes(title="Game Name")
    st.plotly_chart(fig1)
    st.write("💡")

#2
    st.subheader('Top 10 Game Genes/Tags')
    s1['tags'] = s1['tags'].fillna('')
    s1['tag_list'] = s1['tags'].str.split(',')
    tags = s1.explode('tag_list')
    tags['tag_list'] = tags['tag_list'].str.strip()
    tags= tags[tags['tag_list'] != '']
    unique_tags = tags['tag_list'].unique()
    selected_tags = st.multiselect('Select tags to display', options=unique_tags, default=unique_tags[:10])
    filtered_tags = tags[tags['tag_list'].isin(selected_tags)]
    tag_counts = filtered_tags['tag_list'].value_counts()
    fig2 = px.bar(tag_counts[::-1], orientation='h', title='Top 10r Game Genes/Tags')
    fig2.update_layout(yaxis_title='Tags', xaxis_title='Number of Games')
    st.plotly_chart(fig2)
    st.write("🎮")

#3
    st.subheader('Game Rankings Across Different Genres')
    genres = steam4['genre'].unique()
    selected_genre = st.selectbox('Select a genre', genres)
    filtered_data = steam4[steam4['genre'] == selected_genre]
    fig3 = px.box(steam4, x='genre', y='rank', title='Game Rankings Across Different Genres')
    st.plotly_chart(fig3)
    st.write("🎮")

#4
    st.subheader('Top 10 Ranked Games by Genre')
    top_n_rank = st.slider('Select top N ranks', min_value=5, max_value=50, value=10)
    top_ranked_games = steam4.sort_values(by='rank').groupby('genre').head(top_n_rank)
    fig4 = px.bar(top_ranked_games, x='game_name', y='rank', color='genre', title='Top 10 Ranked Games by Genre', 
                labels={'rank':'Rank', 'game_name':'Game Name'}, hover_name='game_name')
    st.plotly_chart(fig4)

#5
    st.subheader('Distribution of Game Prices')
    min_price = s1['price_x'].min()
    max_price = s1['price_x'].max()
    price_range = st.slider('Select price range', float(min_price), float(max_price), (float(min_price), float(max_price)))
    filtered_s1 = s1[(s1['price_x'] >= price_range[0]) & (s1['price_x'] <= price_range[1])]
    fig5 = px.histogram(filtered_s1, x='price_x', nbins=50, title='Distribution of Game Prices')
    fig5.update_xaxes(title='Price ($)')
    fig5.update_yaxes(title='Number of Games')
    st.plotly_chart(fig5)

#6
    st.subheader('Game Releases Over Time')
    steam1['release_date'] = pd.to_datetime(steam1['release_date'], errors='coerce')
    min_date = steam1['release_date'].min()
    max_date = steam1['release_date'].max()
    date_range = st.date_input('Select date range', [min_date, max_date])
    mask = (steam1['release_date'] >= pd.to_datetime(date_range[0])) & (steam1['release_date'] <= pd.to_datetime(date_range[1]))
    release_date_trend = steam1.loc[mask].groupby(steam1.loc[mask]['release_date'].dt.year).size()
    fig6 = px.line(release_date_trend, x=release_date_trend.index, y=release_date_trend.values, title="Game Releases Over Time")
    fig6.update_xaxes(title="Year")
    fig6.update_yaxes(title="Number of Games Released")
    st.plotly_chart(fig6)

#7
    st.subheader('Player Ratings and Review Types Distribution')
    rating_distribution = steam5['overall_player_rating'].value_counts().reset_index()
    rating_distribution.columns = ['overall_player_rating', 'count']
    review_type_counts = steam1['review_type'].value_counts()
    fig7 = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type':'domain'}, {'type':'domain'}]],
        subplot_titles=['Distribution of Overall Player Ratings', 'Distribution of Review Types']
    )

    colors1 = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    fig7.add_trace(
        go.Pie(labels=rating_distribution['overall_player_rating'], values=rating_distribution['count'],
            marker=dict(colors=colors1)),
        row=1, col=1
    )

    fig7.add_trace(
        go.Pie(labels=review_type_counts.index, values=review_type_counts.values,
            marker=dict(colors=colors1)),
        row=1, col=2
    )

    fig7.update_layout(title_text='Player Ratings and Review Types Distribution')
    st.plotly_chart(fig7)

#8 
    st.subheader('Distribution of Publisher Class')
    fig8 = px.pie(steam2, names='publisherclass', title='Distribution of Publisher Classes')
    st.plotly_chart(fig8)
#9
    st.subheader('Avg Playtime vs Review Score by Publisher Class')
    publisher_classes = s1['publisherclass'].unique()
    selected_publisher_classes = st.multiselect('Select Publisher Classes', publisher_classes, default=publisher_classes)
    filtered_s1 = s1[s1['publisherclass'].isin(selected_publisher_classes)]
    fig9 = px.scatter(filtered_s1, x='avgplaytime', y='reviewscore', color='publisherclass', title='Avg Playtime vs Review Score by Publisher Class')
    st.plotly_chart(fig9)
    st.write("1")
#10
    st.subheader('Price vs Review Score')
    min_price = steam2['price'].min()
    max_price = steam2['price'].max()
    min_score = steam2['reviewscore'].min()
    max_score = steam2['reviewscore'].max()
    price_range = st.slider('Select price range', float(min_price), float(max_price), (float(min_price), float(max_price)))
    score_range = st.slider('Select review score range', float(min_score), float(max_score), (float(min_score), float(max_score)))
    filtered_steam2 = steam2[(steam2['price'] >= price_range[0]) & (steam2['price'] <= price_range[1]) & 
                            (steam2['reviewscore'] >= score_range[0]) & (steam2['reviewscore'] <= score_range[1])]
    fig20 = px.scatter(filtered_steam2, x='price', y='reviewscore', title="Price vs Review Score", hover_data=['name'], color='publisherclass')
    fig20.update_xaxes(title="Price (USD)")
    fig20.update_yaxes(title="Review Score")
    st.plotly_chart(fig20)
#11
    st.subheader('3D Scatter Plot of Reviews, Ratings, and Genre')
    x_axis = st.selectbox('Select X-axis', s2.columns, index=s2.columns.get_loc('number_of_english_reviews'))
    y_axis = st.selectbox('Select Y-axis', s2.columns, index=s2.columns.get_loc('overall_player_rating'))
    z_axis = st.selectbox('Select Z-axis', s2.columns, index=s2.columns.get_loc('rank'))
    fig11 = px.scatter_3d(s2, x=x_axis, y=y_axis, z=z_axis, color='genre', title='3D Scatter Plot of Reviews, Ratings, and Genre')
    st.plotly_chart(fig11)

#12
    st.subheader('Correlation Heatmap of Numerical Variables')
    corr_threshold = st.slider('Select correlation coefficient threshold', 0.0, 1.0, 0.5)
    numeric_cols = ['review_no', 'copiessold', 'revenue', 'avgplaytime', 'reviewscore']
    corr_matrix = s1[numeric_cols].corr()
    mask = abs(corr_matrix) >= corr_threshold
    filtered_corr = corr_matrix.where(mask)
    fig12 = px.imshow(filtered_corr, text_auto=True, title='Correlation Heatmap of Numerical Variables')
    st.plotly_chart(fig12)

    st.write("All the visualization offers insights into the datasets, such as the distribution of review counts, prices, genres, and correlations between numerical features. ")

elif page == "WordCloud🌨️":
    st.title("WordCloud🌨️")

    st.markdown("### 🌟 Overview")
    st.markdown("In this section, we use **Word Clouds** to visualize the most frequently occurring words and phrases in various text fields. Word clouds help us uncover hidden patterns and trends in the dataset, offering an intuitive way to explore textual data. 🎨")

    st.markdown("### 🔍 Key Insights")
    st.markdown("- **Tags Word Cloud**: Highlights the most common tags associated with games, showcasing popular genres and features.")
    st.markdown("- **Long Description Word Cloud**: Extracts key terms from the detailed game descriptions, revealing how developers describe their games to potential players.")
    st.markdown("- **Short Description Word Cloud**: Focuses on the shorter promotional descriptions, which provide a concise summary of the game's features and appeal.")

    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import Counter
    import nltk
    from streamlit_echarts import st_pyecharts
    from pyecharts.charts import WordCloud as PyEchartsWordCloud
    from pyecharts import options as opts

    nltk.download('punkt')
    nltk.download('stopwords')

    steam1 = pd.read_csv('steam1.csv')
    steam2 = pd.read_csv('steam2.csv')
    steam4 = pd.read_csv('steam4.csv')
    steam5 = pd.read_csv('steam5.csv')
    s1 = pd.read_csv('s1.csv')
    s2 = pd.read_csv('s2.csv')
    s3 = pd.read_csv('s3.csv')

    st.markdown("#### 🎮 Tags Word Cloud")
    all_tags = ' '.join(s1['tags'].dropna()) 
    wordcloud_tags = WordCloud(width=800, height=400, background_color='white').generate(all_tags)
    wordcloud_array = np.array(wordcloud_tags)
    fig = px.imshow(wordcloud_array, title="Word Cloud of Game Tags", binary_string=True)
    fig.update_xaxes(visible=False)  
    fig.update_yaxes(visible=False) 
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))  
    st.plotly_chart(fig)

    st.markdown("#### 📝 Long Description Word Cloud")
    long_desc_text = ' '.join(s2['long_description'].dropna())
    wordcloud_long_desc = WordCloud(background_color="white", max_words=200).generate(long_desc_text)
    fig = px.imshow(wordcloud_long_desc, title="Word Cloud for Game Long Descriptions")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))  
    st.plotly_chart(fig)

    st.markdown("#### ✨ Short Description Word Cloud")
    short_desc_text = ' '.join(s2['short_description'].dropna())
    wordcloud_short_desc = WordCloud(background_color="white", max_words=200).generate(short_desc_text)
    fig = px.imshow(wordcloud_short_desc, title="Word Cloud for Game Short Descriptions")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))  
    st.plotly_chart(fig)

    st.markdown("### 🔗 Summary")
    st.markdown("- The **Tags Word Cloud** helps identify popular genres or gameplay features.")
    st.markdown("- The **Long Description Word Cloud** provides insights into how games are marketed to players, highlighting key themes.")
    st.markdown("- The **Short Description Word Cloud** emphasizes the concise features that attract players' attention.")
    st.markdown("Word Clouds are a visually appealing way to explore textual data, guiding deeper analyses into player preferences and game characteristics. 🚀")

elif page == "Models🏈":
    st.title("Models🏈")

    st.write("On this page, we showcase the training and evaluation results of multiple machine learning models 🤖📊 and explore a content-based game recommendation system 🎮✨.")
    st.markdown("### 📊 Overview")
    st.markdown("- **Part 1: Data Analysis and Modeling**")
    st.markdown("  - Includes training and evaluation of models such as Linear Regression, Random Forest, and PCA visualizations.")
    st.markdown("- **Part 2: Recommendation System**")
    st.markdown("  - A content-based recommendation system built using advanced feature engineering and model evaluation.")
    st.markdown("### 🔑 Key Features")
    st.markdown("- **Interactive Model Training**: Users can select features and parameters for training.")
    st.markdown("- **Model Evaluation**: Provides insights into model performance using metrics like MSE, R², and visualizations.")
    st.markdown("- **Recommendation System**: Suggests games based on player preferences and game attributes.")

    import pandas as pd
    import numpy as np
    from datetime import datetime
    import re
    import ast
    from collections import Counter
    from scipy.sparse import hstack
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from wordcloud import WordCloud
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score
    )
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
    from sklearn.naive_bayes import MultinomialNB
    import xgboost as xgb
    from imblearn.over_sampling import RandomOverSampler
    import warnings
    warnings.filterwarnings('ignore')


    steam1 = pd.read_csv('steam1.csv')
    steam2 = pd.read_csv('steam2.csv')
    steam4 = pd.read_csv('steam4.csv')
    steam5 = pd.read_csv('steam5.csv')
    s1 = pd.read_csv('s1.csv')
    s2 = pd.read_csv('s2.csv')

    st.subheader('Linear Regression')

    st.write("Select X and Y")
    numeric_cols = s1.select_dtypes(include=['float64', 'int64']).columns.tolist()
    x_var = st.selectbox('X', numeric_cols, index=numeric_cols.index('avgplaytime') if 'avgplaytime' in numeric_cols else 0)
    y_var = st.selectbox('Y', numeric_cols, index=numeric_cols.index('reviewscore') if 'reviewscore' in numeric_cols else 0)

    X = s1[[x_var]].dropna()
    y = s1[y_var][X.index].dropna()
    X = X.loc[y.index]
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test[x_var], y=y_test, mode='markers', name='Real'))
    fig.add_trace(go.Scatter(x=X_test[x_var], y=y_pred, mode='lines', name='Prediction', line=dict(color='red')))
    fig.update_layout(title='LinearRegression', xaxis_title=x_var, yaxis_title=y_var)
    st.plotly_chart(fig)

    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R²: {r2_score(y_test, y_pred):.2f}")

    st.subheader('KDE Analysis')

    st.write("Select factor X for KDE")
    x = st.selectbox('Select X', numeric_cols, key='kde_var')
    fig_kde = px.histogram(s1, x=x, nbins=50, histnorm='density', opacity=0.7, marginal='rug')
    fig_kde.update_layout(title=f'{x}-KDE ', xaxis_title=x, yaxis_title='density')
    st.plotly_chart(fig_kde)

    st.subheader('PCA1')
    st.write("Select Features:")
    pca_vars = st.multiselect('Select', numeric_cols, default=numeric_cols[:5], key='pca_vars')

    if len(pca_vars) >= 2:
        X_pca = s1[pca_vars].dropna()
        X_scaled = StandardScaler().fit_transform(X_pca)
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X_scaled)
        principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, s1[['publisherclass']].iloc[X_pca.index].reset_index(drop=True)], axis=1)
        fig_pca1 = px.scatter(finalDf, x='principal component 1', y='principal component 2', color='publisherclass',
                              title='PCA1', labels={'publisherclass': 'Publisher Class'})
        st.plotly_chart(fig_pca1)
    else:
        st.write("Select Two")

    st.subheader('PCA2')
    if len(pca_vars) >= 3:
        pca_3d = PCA(n_components=3)
        principalComponents_3d = pca_3d.fit_transform(X_scaled)
        principalDf_3d = pd.DataFrame(data=principalComponents_3d,
                                      columns=['principal component 1', 'principal component 2', 'principal component 3'])
        finalDf_3d = pd.concat([principalDf_3d, s1[['publisherclass']].iloc[X_pca.index].reset_index(drop=True)], axis=1)
        fig_pca2 = px.scatter_3d(finalDf_3d, x='principal component 1', y='principal component 2',
                                 z='principal component 3', color='publisherclass',
                                 title='PCA2', labels={'publisherclass': 'Publisher Class'})
        st.plotly_chart(fig_pca2)
    else:
        st.write("Please select three...")

    st.subheader('Random Forest')
    st.write("Select factor:")
    target_var = st.selectbox('Select Target', numeric_cols, index=numeric_cols.index('reviewscore') if 'reviewscore' in numeric_cols else 0, key='rf_target_var')

    feature_vars = st.multiselect('Select Feature', [col for col in numeric_cols if col != target_var], default=['avgplaytime', 'copiessold'], key='rf_feature_vars')

    if feature_vars:
        X_rf = s1[feature_vars].dropna()
        y_rf = s1[target_var][X_rf.index].dropna()
        X_rf = X_rf.loc[y_rf.index]
        y_rf = y_rf.loc[X_rf.index]

        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_rf, y_train_rf)
        y_pred_rf = rf_model.predict(X_test_rf)
        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_vars, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

        fig_rf = px.bar(feature_importance_df, x='feature', y='importance', title='importance')
        st.plotly_chart(fig_rf)

        fig_rf_pred = go.Figure()
        fig_rf_pred.add_trace(go.Scatter(x=y_test_rf, y=y_pred_rf, mode='markers'))
        fig_rf_pred.update_layout(title='Real vs Prediction', xaxis_title='Real', yaxis_title='Prediction')
        st.plotly_chart(fig_rf_pred)

        st.write(f"MSE: {mean_squared_error(y_test_rf, y_pred_rf):.2f}")
        st.write(f"R²: {r2_score(y_test_rf, y_pred_rf):.2f}")
    else:
        st.write("please select one...")

    st.markdown("## Part2. Recommandation System")
    @st.cache_data
    def load_and_clean_data_models():
        dtypes = {
            'name': 'category',
            'short_description': 'string',
            'long_description': 'string',
            'genres': 'string',
            'minimum_system_requirement': 'string',
            'recommend_system_requirement': 'string',
            'release_date': 'string',
            'developer': 'string',
            'publisher': 'string',
            'overall_player_rating': 'string',
            'number_of_reviews_from_purchased_people': 'string',
            'number_of_english_reviews': 'string',
            'link': 'string'
        }
        df_model = pd.read_csv('steam5.csv', dtype=dtypes, low_memory=True)

        def parse_list_field(field):
            try:
                return ast.literal_eval(field)
            except (ValueError, SyntaxError):
                return []

        df_model['genres'] = df_model['genres'].apply(parse_list_field)
        df_model['developer'] = df_model['developer'].apply(parse_list_field)
        df_model['publisher'] = df_model['publisher'].apply(parse_list_field)

        def parse_number_field(x):
            if pd.isnull(x):
                return 0
            x = re.sub(r'[\(\),]', '', x)
            x = x.replace(',', '')
            try:
                return int(x)
            except ValueError:
                return 0

        df_model['number_of_reviews_from_purchased_people'] = df_model['number_of_reviews_from_purchased_people'].apply(parse_number_field)
        df_model['number_of_english_reviews'] = df_model['number_of_english_reviews'].apply(parse_number_field)

        def parse_date(date_str):
            try:
                return datetime.strptime(date_str, '%d %b, %Y')
            except ValueError:
                try:
                    return datetime.strptime(date_str, '%b %Y')
                except ValueError:
                    return pd.NaT

        df_model['release_date'] = df_model['release_date'].apply(parse_date)

        return df_model

    df_model = load_and_clean_data_models()
    @st.cache_resource
    def feature_engineering_and_train_models(df):
        valid_ratings = [
            'Overwhelmingly Positive', 'Very Positive', 'Mostly Positive', 'Positive',
            'Mixed', 'Negative', 'Mostly Negative', 'Very Negative', 'Overwhelmingly Negative'
        ]
        df = df[df['overall_player_rating'].isin(valid_ratings)]

        rating_mapping = {
            'Overwhelmingly Positive': 'Positive',
            'Very Positive': 'Positive',
            'Mostly Positive': 'Positive',
            'Positive': 'Positive',
            'Mixed': 'Mixed',
            'Mostly Negative': 'Negative',
            'Negative': 'Negative',
            'Very Negative': 'Negative',
            'Overwhelmingly Negative': 'Negative'
        }

        df['overall_player_rating'] = df['overall_player_rating'].map(rating_mapping)

        le_rating = LabelEncoder()
        df['overall_player_rating_encoded'] = le_rating.fit_transform(df['overall_player_rating'])

        df['description'] = df['short_description'].fillna('') + ' ' + df['long_description'].fillna('')

        max_tfidf_features = 5000  
        tfidf = TfidfVectorizer(max_features=max_tfidf_features, stop_words='english', ngram_range=(1, 2))
        text_features = tfidf.fit_transform(df['description'])

        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(df['genres'])

        numeric_features = df[['number_of_reviews_from_purchased_people', 'number_of_english_reviews']].fillna(0)

        final_features = hstack([text_features, genres_encoded, numeric_features])

        X = final_features
        y = df['overall_player_rating_encoded']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

        k_best_features = 1000  
        selector = SelectKBest(chi2, k=k_best_features)
        X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
        X_test_selected = selector.transform(X_test)

        models = {
            'LinearSVC': LinearSVC(max_iter=5000, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1),
            'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'MultinomialNB': MultinomialNB(),
            'LogisticRegression': LogisticRegression(max_iter=5000, class_weight='balanced', n_jobs=-1)
        }

        model_results = {}

        for name, model in models.items():
            model.fit(X_train_selected, y_train_resampled)
            y_pred = model.predict(X_test_selected)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le_rating.classes_, zero_division=0, output_dict=True)
            model_results[name] = {
                'model': model,
                'accuracy': acc,
                'report': report
            }

        svd = TruncatedSVD(n_components=100, random_state=42)
        features_reduced = svd.fit_transform(X)

        nn_model = NearestNeighbors(n_neighbors=6, algorithm='auto', n_jobs=-1)
        nn_model.fit(features_reduced)

        return le_rating, tfidf, mlb, X, df, models, model_results, svd, nn_model

    le_rating, tfidf, mlb, X_all, df_all, models, model_results, svd, nn_model = feature_engineering_and_train_models(df_model)

    st.subheader("Model Evaluation Results")

    model_names = list(models.keys())
    selected_model = st.selectbox("Select A Model: ", model_names)

    if selected_model:
        st.write(f"**{selected_model} Model Accuracy:{model_results[selected_model]['accuracy']:.4f}**")
        report_df = pd.DataFrame(model_results[selected_model]['report']).transpose()
        st.dataframe(report_df)
        
    rating_counts = df_model['overall_player_rating'].value_counts()

    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={"x": "Player Rating", "y": "Count"},
        title="Distribution of Player Ratings"
    )
    fig.update_layout(xaxis_title="Player Rating", yaxis_title="Count")
    st.plotly_chart(fig)


    all_genres = [genre for sublist in df_model['genres'] for genre in sublist]
    genres_count = pd.Series(all_genres).value_counts().nlargest(10)
    fig = px.bar(
        x=genres_count.index,
        y=genres_count.values,
        labels={"x": "Game Genre", "y": "Count"},
        title="Top 10 Most Popular Game Genres"
    )
    fig.update_layout(xaxis_title="Game Genre", yaxis_title="Count")
    st.plotly_chart(fig)

    df_model['release_year'] = df_model['release_date'].dt.year
    fig = px.histogram(
        df_model, 
        x="release_year", 
        title="Yearly Game Releases",
        labels={"release_year": "Year", "count": "Number of Games"},
        nbins=len(df_model['release_year'].dropna().unique())
    )
    fig.update_layout(xaxis_title="Release Year", yaxis_title="Number of Games")
    st.plotly_chart(fig)

    fig = px.box(
        df_model,
        x="overall_player_rating",
        y="number_of_reviews_from_purchased_people",
        points="all",
        title="Number of Reviews by Player Ratings",
        labels={"overall_player_rating": "Player Rating", "number_of_reviews_from_purchased_people": "Number of Reviews"}
    )
    fig.update_layout(yaxis_type="log", xaxis_title="Player Rating", yaxis_title="Number of Reviews (Log Scale)")
    st.plotly_chart(fig)

    dtypes = {
        'name': 'category',
        'short_description': 'string',
        'long_description': 'string',
        'genres': 'string',
        'minimum_system_requirement': 'string',
        'recommend_system_requirement': 'string',
        'release_date': 'string',
        'developer': 'string',
        'publisher': 'string',
        'overall_player_rating': 'string',
        'number_of_reviews_from_purchased_people': 'string',
        'number_of_english_reviews': 'string',
        'link': 'string'
    }

    df_model['description'] = df_model['short_description'].fillna('') + ' ' + df_model['long_description'].fillna('')  
    text = " ".join(df_model['description'].fillna(""))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig = px.imshow(wordcloud, title="Word Cloud for Game Descriptions")
    fig.update_xaxes(visible=False)
    st.plotly_chart(fig)

elif page == "Recommendation🎯":
    st.title("Recommendation🎯")

    import pandas as pd
    import numpy as np
    import re
    from datetime import datetime
    import ast
    from collections import Counter
    from scipy.sparse import hstack
    import streamlit as st
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, chi2
    from imblearn.over_sampling import RandomOverSampler
    import warnings
    warnings.filterwarnings('ignore')
    @st.cache_data
    def load_and_clean_data():
        dtypes = {
            'name': 'category',
            'short_description': 'string',
            'long_description': 'string',
            'genres': 'string',
            'minimum_system_requirement': 'string',
            'recommend_system_requirement': 'string',
            'release_date': 'string',
            'developer': 'string',
            'publisher': 'string',
            'overall_player_rating': 'string',
            'number_of_reviews_from_purchased_people': 'string',
            'number_of_english_reviews': 'string',
            'link': 'string'
        }
        df_rec = pd.read_csv('steam5.csv', dtype=dtypes, low_memory=True)
        return df_rec

    df_rec = load_and_clean_data()

    def clean_data(df):
        def parse_list_field(field):
            try:
                return ast.literal_eval(field)
            except (ValueError, SyntaxError):
                return []

        df['genres'] = df['genres'].apply(parse_list_field)
        df['developer'] = df['developer'].apply(parse_list_field)
        df['publisher'] = df['publisher'].apply(parse_list_field)

        def parse_number_field(x):
            if pd.isnull(x):
                return 0
            x = re.sub(r'[\(\),]', '', x)
            x = x.replace(',', '')
            try:
                return int(x)
            except ValueError:
                return 0

        df['number_of_reviews_from_purchased_people'] = df['number_of_reviews_from_purchased_people'].apply(parse_number_field)
        df['number_of_english_reviews'] = df['number_of_english_reviews'].apply(parse_number_field)

        def parse_date(date_str):
            try:
                return datetime.strptime(date_str, '%d %b, %Y')
            except ValueError:
                try:
                    return datetime.strptime(date_str, '%b %Y')
                except ValueError:
                    return pd.NaT

        df['release_date'] = df['release_date'].apply(parse_date)
        
        # Add the 'description' column here
        df['description'] = df['short_description'].fillna('') + ' ' + df['long_description'].fillna('')

        return df

    df_rec = clean_data(df_rec)

    @st.cache_resource
    def feature_engineering_and_train_model(df):
        valid_ratings = [
            'Overwhelmingly Positive', 'Very Positive', 'Mostly Positive', 'Positive',
            'Mixed', 'Negative', 'Mostly Negative', 'Very Negative', 'Overwhelmingly Negative'
        ]
        df = df[df['overall_player_rating'].isin(valid_ratings)]

        rating_mapping = {
            'Overwhelmingly Positive': 'Positive',
            'Very Positive': 'Positive',
            'Mostly Positive': 'Positive',
            'Positive': 'Positive',
            'Mixed': 'Mixed',
            'Mostly Negative': 'Negative',
            'Negative': 'Negative',
            'Very Negative': 'Negative',
            'Overwhelmingly Negative': 'Negative'
        }
        df['overall_player_rating'] = df['overall_player_rating'].map(rating_mapping)

        # The 'description' column is already added in clean_data, no need to add it here
        # df['description'] = df['short_description'].fillna('') + ' ' + df['long_description'].fillna('')

        tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        tfidf.fit(df['description'])

        mlb_genres = MultiLabelBinarizer()
        mlb_genres.fit(df['genres'])
        mlb_developers = MultiLabelBinarizer()
        mlb_developers.fit(df['developer'])
        mlb_publishers = MultiLabelBinarizer()
        mlb_publishers.fit(df['publisher'])

        le_rating = LabelEncoder()
        df['overall_player_rating_encoded'] = le_rating.fit_transform(df['overall_player_rating'])

        text_features = tfidf.transform(df['description'])
        genres_encoded = mlb_genres.transform(df['genres'])
        developers_encoded = mlb_developers.transform(df['developer'])
        publishers_encoded = mlb_publishers.transform(df['publisher'])
        numeric_features = df[['number_of_reviews_from_purchased_people', 'number_of_english_reviews']].fillna(0)
        final_features = hstack([text_features, genres_encoded, developers_encoded, publishers_encoded, numeric_features])

        X = final_features
        y = df['overall_player_rating_encoded']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

        selector = SelectKBest(chi2, k=1000)
        X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)

        rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
        rf_model.fit(X_train_selected, y_train_resampled)

        return rf_model, selector, le_rating, tfidf, mlb_genres, mlb_developers, mlb_publishers

    rf_model_rec, selector_rec, le_rating_rec, tfidf_rec, mlb_genres_rec, mlb_developers_rec, mlb_publishers_rec = feature_engineering_and_train_model(df_rec)

    def recommend_games_rf(genres_list, developer_list, publisher_list):
        mask = np.ones(len(df_rec), dtype=bool)
        if genres_list:
            genres_mask = df_rec['genres'].apply(lambda x: any(genre in x for genre in genres_list))
            mask = mask & genres_mask
        if developer_list:
            dev_mask = df_rec['developer'].apply(lambda x: any(dev in x for dev in developer_list))
            mask = mask & dev_mask
        if publisher_list:
            pub_mask = df_rec['publisher'].apply(lambda x: any(pub in x for pub in publisher_list))
            mask = mask & pub_mask

        filtered_df = df_rec[mask]
        if filtered_df.empty:
            st.write("No games found with the specified criteria.")
            return

        descriptions = filtered_df['description']
        text_features = tfidf_rec.transform(descriptions)
        genres_features = mlb_genres_rec.transform(filtered_df['genres'])
        developers_features = mlb_developers_rec.transform(filtered_df['developer'])
        publishers_features = mlb_publishers_rec.transform(filtered_df['publisher'])
        numeric_features = filtered_df[['number_of_reviews_from_purchased_people', 'number_of_english_reviews']].fillna(0)

        game_features = hstack([text_features, genres_features, developers_features, publishers_features, numeric_features])
        game_features_selected = selector_rec.transform(game_features)

        predicted_ratings = rf_model_rec.predict(game_features_selected)
        filtered_df['predicted_rating'] = predicted_ratings
        filtered_df['predicted_rating_label'] = le_rating_rec.inverse_transform(predicted_ratings)

        recommended_games = filtered_df.sort_values(['predicted_rating', 'number_of_reviews_from_purchased_people'], ascending=[False, False])

        st.write("Recommended Games🕹️:")
        for idx, row in recommended_games.head(10).iterrows():
            st.write(f"**Name🎉**: {row['name']}")
            st.write(f"**Predicted Rating🎢**: {row['predicted_rating_label']}")
            st.write(f"**Genres**: {', '.join(row['genres'])}")
            st.write(f"**Developer**: {', '.join(row['developer'])}")
            st.write(f"**Publisher**: {', '.join(row['publisher'])}")
            st.write(f"**Number of Reviews🥠**: {row['number_of_reviews_from_purchased_people']}")
            st.write("---")

    st.markdown("""
    ### 💫Welcome to the Steam Game Recommendation System!

    💗 **This app helps you find games based on your preferences:**
                
    💗 **Select genres you're interested in**
                
    💗 **Choose specific developers**
                
    💗 **Pick preferred publishers**

    🖱️**Click 'Get Recommendations' to find your perfect games!**
    """)

    all_genres = sorted(set([genre for sublist in df_rec['genres'] for genre in sublist]))
    all_developers = sorted(set([dev for sublist in df_rec['developer'] for dev in sublist]))
    all_publishers = sorted(set([pub for sublist in df_rec['publisher'] for pub in sublist]))

    st.sidebar.header("🎮 Game Preference Filters")
    selected_genres = st.sidebar.multiselect("Select Genres", all_genres)
    selected_developers = st.sidebar.multiselect("Select Developers", all_developers)
    selected_publishers = st.sidebar.multiselect("Select Publishers", all_publishers)

    if st.sidebar.button("🚀 Get Recommendations"):
        recommend_games_rf(selected_genres, selected_developers, selected_publishers)

elif page == "Conclusion🍩":
    st.title("Conclusion🍩")
    st.markdown("""
    This project leverages Steam game data to develop a comprehensive system that integrates data analysis, visualization, and machine learning modeling. Here are the key highlights and outcomes:
    """)

    st.markdown("### 📊 Data Analysis and Exploration")
    st.markdown("""
    1. **Data Cleaning and Preprocessing**:
    - Processed datasets from multiple sources, including game descriptions, player ratings, and sales data, to ensure data quality and consistency.
    - Addressed missing values (e.g., using MICE imputation), handled outliers, and standardized formats for seamless analysis.
    - Engineered new features like `review_density` and `price_per_hour` to enhance analytical depth.

    2. **Exploratory Data Analysis (EDA)**:
    - Interactive visualizations showcased trends in game genres, tags, and ratings, along with the relationship between price and sales volume.
    - Word clouds revealed popular themes in game descriptions and tags, providing insights into player preferences and market trends.
    """)

    st.markdown("### 🤖 Machine Learning Modeling")
    st.markdown("""
    1. **Model Evaluation and Comparison**:
    - Trained and evaluated multiple models (e.g., Linear Regression, Random Forest, and PCA), assessing performance using metrics such as MSE and R².
    - Conducted feature importance analysis to identify critical factors influencing game ratings and sales.

    2. **Recommendation System Development**:
    - Built a content-based recommendation system combining TF-IDF textual features and game tags to provide personalized game suggestions.
    - Integrated visualizations of model evaluation and recommendation results to enhance user engagement.
    """)

    st.markdown("### 🎮 System Features and User Experience")
    st.markdown("""
    - **Multi-functional Pages**:
    - Structured into modules for data display, EDA, modeling, and recommendation system, allowing users to explore and interact dynamically.
    - **Interactive Interface**:
    - Enables users to select variables, adjust parameters, and view real-time results, showcasing the power of data-driven analysis.
    - **Scalability and Practicality**:
    - Designed to integrate new data effortlessly, ensuring broad applicability and robust functionality.
    """)

    st.markdown("### Key Achievements 🚀")
    st.markdown("""
    - Identified critical factors contributing to game success, such as tags, pricing strategies, and player ratings.
    - Demonstrated the potential of data science in enhancing user experience and driving business value through personalized recommendations.
    - Delivered a complete workflow of data analysis and modeling, highlighting the practical application of data-driven decision-making.
    """)