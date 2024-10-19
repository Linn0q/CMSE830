# CMSE830-Project

# **Video Game Sentiment Analysis and Rating Prediction Based on Steam User Ratings**

---

## **Project Overview**
This project explores video game ratings and user reviews on **Steam**, focusing on sentiment analysis and rating prediction. The project is built using **Streamlit** and features interactive visualizations, word clouds, and machine learning models.

## **Key Features**

1. **Homepage**: Introduces the project.

2. **Data Page**: Displays the datasets used in the project.

3. **Gallery Page**: Shows exploratory data analysis (EDA) work.

4. **WordCloud Page**: Generates word clouds based on keywords.

5. **Analysis Page**: Features the initial analysis and model building.
   - **Linear Regression**: Analyzes the playtime and review score relationship.
   - **KDE Analysis**: Shows kernel density estimation for selected variables.
   - **PCA**: Performs Principal Component Analysis to reduce dimensionality.
   - **Random Forest**: Builds a random forest model for prediction and displays feature importance.

6. **Conclusion Page**: Summarizes the project's progress and future plans.
   - Review the completed tasks and outline the upcoming work, including model optimization and advanced analysis.

## **Installation and Setup**

### **Prerequisites**
- Python 
- Streamlit
- Pandas
- Seaborn
- Plotly
- Scikit-learn
- NLTK
- Matplotlib
- WordCloud
- VADER Sentiment (nltk)

## **Project Structure**

```bash
├── app.py                # Main Streamlit app
├── steam1.csv            # Steam dataset 1
├── steam2.csv            # Steam dataset 2
├── steam4.csv            # Additional dataset
├── steam5.csv            # Additional dataset
├── s1.csv                # Sentiment analysis dataset
├── s2.csv                # Sentiment analysis dataset
├── README.md             # Project README
└── .streamlit
    └── config.toml       # Theme configuration for Streamlit
```

## **How to Use the App**

1. **Homepage🎮**:
   - Introduction to the project, featuring an overview of the analysis and goals.
2. **Data🌈**:
   - Explore the datasets and view the cleaning process.
3. **Gallery🌷**:
   - Navigate through various interactive visualizations that offer insights into the dataset.
4. **WordCloud🌨️**:
   - Visualize word clouds based on Steam user reviews and game descriptions.
5. **Analysis🏈**:
   - Explore the initial analysis, including regression and PCA.
6. **Conclusion🍩**:
   - Review the progress made and future plans for the project.

## **Future Work**
1. **Modeling and Prediction**: Build more models to predict game ratings based on user sentiment and game features.
2. **Multivariate Analysis**: Explore the interactions between variables more deeply.
3. **App Optimization**: Improve the Streamlit app for better user interaction and performance.

## **Contributing**

Feel free to contribute to this project by submitting a pull request or opening an issue.

## **Contact**

For any questions or feedback, feel free to contact me at wangj324@msu.edu

---
