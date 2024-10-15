import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Streamlit app title
st.title('Steam Games Data Analysis with MICE Imputation')

# Load datasets (You can provide local file paths or use file upload feature in Streamlit)
@st.cache
def load_data():
    steam1_df = pd.read_csv('steam1.csv')
    steam2_df = pd.read_csv('steam2.csv')
    steam4_df = pd.read_csv('steam4.csv')
    steam5_df = pd.read_csv('steam5.csv')
    return steam1_df, steam2_df, steam4_df, steam5_df

steam1_df, steam2_df, steam4_df, steam5_df = load_data()

# Preprocessing
steam1_df['Release_date'] = pd.to_datetime(steam1_df['Release_date'], format='%Y-%m-%d', errors='coerce')
steam2_df['releaseDate'] = pd.to_datetime(steam2_df['releaseDate'], format='%d-%m-%Y', errors='coerce')
steam5_df['release_date'] = pd.to_datetime(steam5_df['release_date'], format='%d %b, %Y', errors='coerce')

# Clean game names for merging
steam1_df['Name'] = steam1_df['Name'].str.lower().str.strip()
steam2_df['name'] = steam2_df['name'].str.lower().str.strip()
steam4_df['game_name'] = steam4_df['game_name'].str.lower().str.strip()
steam5_df['name'] = steam5_df['name'].str.lower().str.strip()

# Merge datasets
merged_df_1 = pd.merge(steam1_df, steam2_df, left_on='Name', right_on='name', how='outer')
merged_df_2 = pd.merge(merged_df_1, steam4_df, left_on='Name', right_on='game_name', how='outer')
final_merged_df = pd.merge(merged_df_2, steam5_df, left_on='Name', right_on='name', how='outer')

# Visualize missing data before imputation
st.subheader('Missing Data Before MICE Imputation')
sns.heatmap(final_merged_df.isnull(), cbar=False, cmap='viridis')
st.pyplot(plt.gcf())

# MICE Imputation
numerical_cols = final_merged_df.select_dtypes(include=['float64', 'int64']).columns
mice_imputer = IterativeImputer(max_iter=10, random_state=0)
final_merged_df[numerical_cols] = mice_imputer.fit_transform(final_merged_df[numerical_cols])

# Visualize missing data after imputation
st.subheader('Missing Data After MICE Imputation')
sns.heatmap(final_merged_df.isnull(), cbar=False, cmap='viridis')
st.pyplot(plt.gcf())

# Correlation Matrix
st.subheader('Correlation Matrix After MICE Imputation')
sns.heatmap(final_merged_df[numerical_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
st.pyplot(plt.gcf())
