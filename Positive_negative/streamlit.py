import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder




# Set page title
st.set_page_config(page_title="Fraud Detection App")

# Load CSV files
df1 = pd.read_csv("/Users/sobhione/Documents/JENSENS-SCHOOL/ML-Supervised-Learning/ML-Supervised-learning-2/Positive_negative/dfv3-1.csv")
df2 = pd.read_csv("/Users/sobhione/Documents/JENSENS-SCHOOL/ML-Supervised-Learning/ML-Supervised-learning-2/Positive_negative/dfv3-2.csv")

df = pd.concat([df1, df2], axis=0)

# Display dataframe info
st.subheader('DataFrame Info')
st.write(df.info())

# Drop unwanted columns
df.drop(['Unnamed: 0', 'zip', 'trans_num', 'city', 'street'], axis=1, inplace=True)

# Display updated dataframe
st.subheader('Updated DataFrame')
st.write(df.head())

# Check for missing values
st.subheader('Missing Values')
st.write(df.isnull().sum())

# Count the number of duplicate rows
st.subheader('Duplicate Rows')
st.write(df.duplicated().sum())

# Consider only fraud data
fraud_data = df[df.is_fraud == 1]

st.subheader('Fraud Data Count')
st.write(df.is_fraud.value_counts())

st.subheader('Fraud Data Percentage')
st.write(df.is_fraud.value_counts(normalize=True) * 100)

# plot where you compare gender vs count taking only fraud data

st.subheader('Gender vs Count (Fraud Data)')
colors = ['#c6e2ff', 'pink']
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=df.gender, palette=colors)
plt.title('Gender Vs Counts')

st.set_option('deprecation.showPyplotGlobalUse', False)


# Display the plot
st.pyplot()

# Display fraud data in every state
st.subheader('Fraud Data in Every State')
plt.figure(figsize=(10, 15))
sns.countplot(y='state', data=fraud_data, order=fraud_data.state.value_counts().index)
plt.title('Fraud Data in Every State')

# Display the plot
st.pyplot()
# Consider only fraud data
fraud_data = df[df.is_fraud == 1]

# Display counts of fraud data
st.write("Counts of Fraud Data:")
st.write(fraud_data["is_fraud"].value_counts())

# Display percentages of fraud data
st.write("Fraud Data in Percentage:")
st.write(fraud_data["is_fraud"].value_counts(normalize=True) * 100)

# Create countplot of categories
plt.figure(figsize=(10, 5))
sns.countplot(y=df.category, order=df.category.value_counts().index)
plt.title("Countplot of Categories")
plt.xlabel("Count")
plt.ylabel("Category")
st.pyplot()


# Set the color palette
colors = ['#c6e2ff', 'pink']

# Create the first plot using the entire DataFrame
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=df.gender, palette=colors)
plt.title('Gender Vs Counts')

# Create the second plot using only the fraud data
plt.subplot(1, 2, 2)
sns.countplot(x='gender', data=fraud_data, palette=[colors[1], '#c6e2ff'])
plt.title('Fraud Data')

# Display the plots in the Streamlit app
st.pyplot()

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Create the histogram plot for fraud amount vs. percent
plt.figure(figsize=(10, 5))
plot = sns.histplot(x='amt', data=fraud_data, bins=20, stat='percent', kde=True)

# Format y-axis labels with percentage sign
plot.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(fraud_data)))

# Set plot title and labels
plt.title('Fraud Amount vs. Percent')
plt.xlabel('Amount')
plt.ylabel('Percent')

# Display the plot
st.pyplot()



# Encoding
df_copy = df.copy()

# Label Encoding
le = LabelEncoder()

# Apply label encoding to the desired column(s)
df_copy['encoded_column'] = le.fit_transform(df_copy['column_to_encode'])

# Display the encoded data
st.write(df_copy)