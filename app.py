import streamlit as st

page = st.sidebar.selectbox("Select Page🛗", ["Homepage🎮","Data🌈", "Gallery🌷","WordCloud🌨️","Analysis🏈","Conclusion🍩"])

if page == "Homepage🎮":
    st.title("Homepage🎮")
    st.write("")
    st.image("https://steamuserimages-a.akamaihd.net/ugc/922549154591526002/87EFBCBF9BEBF5F42CD6E9DADB9EE0CCB79A6E38/?imw=5000&imh=5000&ima=fit&impolicy=Letterbox&imcolor=%23000000&letterbox=false", use_column_width=True)
    st.markdown("# Hello!🌈")
    st.markdown("### Welcome to my CMSE-830 project!")
    #st.markdown(" Do you like games? ")
    #st.markdown("### Let's Explore Data in Games！")
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
    st.markdown('- In *Page Data🌈*, I will display my datasets and the detailed cleaning processing is on my Github.')
    st.markdown('- In *Page Gallery🌷*, I will display my EDA (Exploratory Data Analysis) work.Through a series of visualizations, we can find  ')
    st.markdown('- In *Page WordCloud🌨️*, I will display my wordcloud analysis. We can discover the hidden trends and patterns in the words.')
    st.markdown('- In *Page Analysis🏈*, I will display my initial analysis from previous steps, and I will focus more on this part in my rest semester.')
    st.markdown('- In *Page Conclusion🍩*, I will summarize the results of my analysis and we can discuss together!')
    st.write("")
    st.markdown('**Join me on this journey and to understand better the art and science behind games🎉**')
    st.markdown('**Enjoy!🍕🍔🍟🍨🍪🍫🍬🍰**')
  
elif page == "Data🌈":
    st.title("Data")

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.experimental import enable_iterative_imputer  # Enable if using older scikit-learn
    from sklearn.impute import IterativeImputer

    steam1 = pd.read_csv('steam1.csv')
    steam2 = pd.read_csv('steam2.csv')
    steam4 = pd.read_csv('steam4.csv')
    steam5 = pd.read_csv('steam5.csv')

    st.write("About Datasets")
    st.dataframe(steam1.head())
    st.dataframe(steam2.head())
    st.dataframe(steam4.head())
    st.dataframe(steam5.head())

elif page == "Gallery🌷":
    st.title("Gallery🌷")
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
    from textblob import TextBlob
    from sklearn.impute import SimpleImputer
    import streamlit as st
    # Load datasets
    steam1 = pd.read_csv('steam1.csv')
    steam2 = pd.read_csv('steam2.csv')
    #steam3 = pd.read_csv('steam3.csv', low_memory=False)
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
    steam3 = pd.read_csv('steam3.csv', low_memory=False)
    steam4 = pd.read_csv('steam4.csv')
    steam5 = pd.read_csv('steam5.csv')
    s1 = pd.read_csv('s1.csv')
    s2 = pd.read_csv('s2.csv')

  
    def create_wordcloud(common_words, title):
        wordcloud = (
            PyEchartsWordCloud(init_opts=opts.InitOpts(width='800px', height='600px'))
            .add("", common_words, word_size_range=[20, 100], shape='circle')  
            .set_global_opts(title_opts=opts.TitleOpts(title=title))
        )
        return wordcloud

    st.subheader('Word Cloud of Game Descriptions')

    text = ' '.join(steam3['review'].dropna())


    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Game Descriptions')
    st.pyplot(plt)

    st.subheader('Word Cloud of Game Tags')


    all_tags = ' '.join(s1['tags'].dropna())

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_tags)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Game Tags")
    st.pyplot(plt)

    st.subheader('Word Cloud for Game Long Descriptions')

    long_desc_text = ' '.join(s2['long_description'].dropna())

    wordcloud = WordCloud(background_color="white", max_words=200).generate(long_desc_text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Word Cloud for Game Descriptions')
    st.pyplot(plt)

    st.subheader('Interactive Word Cloud of Game Descriptions')

    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]

    word_counts = Counter(words)
    common_words = word_counts.most_common(200)

    wordcloud_chart = create_wordcloud(common_words, 'Interactive Word Cloud of Game Descriptions')
    st_pyecharts(wordcloud_chart)

    st.subheader('Interactive Word Cloud of Game Tags')

    tags = ','.join(s1['tags'].dropna()).split(',')
    tags = [tag.strip().lower() for tag in tags if tag.strip() != '']
    tag_counts = Counter(tags)
    common_tags = tag_counts.most_common(200)

    wordcloud_chart_tags = create_wordcloud(common_tags, 'Interactive Word Cloud of Game Tags')
    st_pyecharts(wordcloud_chart_tags)

    st.subheader('Interactive Word Cloud for Game Long Descriptions')
    long_desc_words = word_tokenize(long_desc_text.lower())
    long_desc_words = [word for word in long_desc_words if word.isalpha() and word not in stop_words]
    long_desc_word_counts = Counter(long_desc_words)
    common_long_desc_words = long_desc_word_counts.most_common(200)
    wordcloud_chart_long_desc = create_wordcloud(common_long_desc_words, 'Interactive Word Cloud for Game Descriptions')
    st_pyecharts(wordcloud_chart_long_desc)

elif page == "Analysis🏈":
    st.title("Analysis🏈")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import plotly.express as px
    import plotly.graph_objects as go

    steam1 = pd.read_csv('steam1.csv')
    steam2 = pd.read_csv('steam2.csv')
    steam3 = pd.read_csv('steam3.csv', low_memory=False)
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

elif page == "Conclusion🍩":
    st.title("Conclusion🍩")
    st.markdown("*Summary for midterm*🍩")
    st.markdown('**So far, I have completed the following tasks for the project:**')
    st.markdown('1. Data Collection and Cleaning')
    st.markdown('2. Exploratory Data Analysis')
    st.markdown('3. Sentiment Analysis')
    st.markdown('4. Initial Visualizations')
    st.markdown('**Here is my future plan:**')
    st.markdown('1. Modeling and prediction')
    st.markdown('2. Deeper multivariate analysis')
    st.markdown('3. Optimization of my Streamlit App')
    





