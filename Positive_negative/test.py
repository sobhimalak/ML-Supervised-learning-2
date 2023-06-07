import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Here there are 2 csv files one for training and other for testing

df1=pd.read_csv("/Users/sobhione/Documents/JENSENS-SCHOOL/ML-Supervised-Learning/ML-Supervised-learning-2/Positive_negative/dfv3-1.csv")
df2=pd.read_csv("/Users/sobhione/Documents/JENSENS-SCHOOL/ML-Supervised-Learning/ML-Supervised-learning-2/Positive_negative/dfv3-2.csv")

df=pd.concat([df1,df2],axis=0)

print('df1:',df1.shape)
print('df2:',df2.shape)
print('df is now Concatenated:',df.shape)
df.head()

df.info()

# print trans_date_trans_time datatype

#changing datatypes
df.trans_date_trans_time=pd.to_datetime(df.trans_date_trans_time)
df.dob=pd.to_datetime(df.dob)

df.describe()

df.describe(include=['object']).T

# drop unwanted columns

df.drop(['Unnamed: 0', 'zip', 'trans_num','city','street'],axis=1,inplace=True)

# Extract year, month,dayname and hour from trans_date_trans_time

df['year']=df.trans_date_trans_time.dt.year
df['month']=df.trans_date_trans_time.dt.month
df['day_name']=df.trans_date_trans_time.dt.day_name()
df['hour']=df.trans_date_trans_time.dt.hour

# Extract age from dob
#df["age"]=df.year-df["dob"].dt.year

df['age']=df.dob.apply(lambda x:(pd.Timestamp.now().year - x.year))
    
#print
df.head()

#  Extract distance from customer to merchant
df["lat_dist_cust_merch"]=(df["lat"]-df["merch_lat"]).abs() 
df.head()

df["long_dist_cust_merch"]=(df['long']-df['merch_long']).abs()
df.head()

# drop columns lat, merch_lat,long, merch_long, dob and trans_date_trans_time

df.drop(['lat','merch_lat','long','merch_long','dob','trans_date_trans_time'], axis=1, inplace=True)
df.head()

df.isnull().sum()

# count the number of duplicate rows
df.duplicated().sum()

## consider only fraud data
fraud_data=df[df.is_fraud==1]

print(df.is_fraud.value_counts())
print("\nIn percentage(%)\n")
print(df.is_fraud.value_counts(normalize=True)*100)

sns.countplot(y=df.category,order=df.category.value_counts().index)

# create a plot where you compare gender vs count taking only fraud data

 #Set the color palette
colors = ['#c6e2ff', 'pink']

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(x=df.gender,palette=colors)
plt.title('Gender Vs Counts')
 
# create another plot
plt.subplot(1,2,2)
sns.countplot(x='gender', data=fraud_data ,palette=[colors[1], '#c6e2ff']) # using only fraud data
plt.title('Fraud Data')

import matplotlib.ticker as ticker

# histplot for fraud amount vs percent

plot = sns.histplot(x='amt',data=fraud_data,bins= 20,stat='percent' ,kde=True) # using only fraud data

# Format y-axis labels with percentage sign
plot.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(fraud_data)))

plt.figure(figsize=(10,15))
# plot fraud data for state
sns.countplot(y='state',data=fraud_data,order=fraud_data.state.value_counts().index)
# title
plt.title('Fraud Data in every State')

# plot fraud data for day_name

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.countplot(x='day_name',data=fraud_data)
plt.title('Day Vs Counts')
plt.xlabel('Days of the week')


plt.subplots_adjust(hspace=0.5)  # Add space between subplots


plt.subplot(2,1,2)
sns.countplot(x='month',data=fraud_data)
plt.title('Month Vs Counts')


# plot fraud data for year

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
#sns.countplot(y='year', data=fraud_data, order=fraud_data['year'].value_counts().index, palette='Blues')
sns.countplot(x='year',data=fraud_data, palette='Greens')
plt.title('Fraud Data - By Year')

plt.subplot(1,2,2)
sns.countplot(x='hour',data=fraud_data)
plt.title('Fraud Data - By Hour')


# plot age

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.histplot(df.age,kde=True)
plt.title("Age Distribution in Overall Data")

plt.subplots_adjust(hspace=0.5)  # Add space between subplots

#plot age with fraud_data


plt.subplot(2,1,2)
ax = sns.histplot(fraud_data.age,kde=True,color='white')
ax.set_facecolor('#14ebe5')  # Set background color to light gray

plt.title("Age Distribution in Fraud data")


#  generate word cloud

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt



# Create a wordcloud object
wordcloud = WordCloud(width=1000, height=500, background_color='black', stopwords=STOPWORDS, min_font_size=10).generate(str(df['category']))

# Plot the wordcloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

#plot lat_dist_cust_merch

data = df.query('is_fraud == 1')

plt.figure(figsize=(10, 6))
sns.histplot(x='lat_dist_cust_merch', data=data, kde=True)
plt.title('Latitude Distance between Customer and Merchant (Fraud Data)')
plt.xlabel('Latitude Distance')
plt.ylabel('Count')


#plot long_dist_cust_merch

plt.figure(figsize=(10, 6))
sns.histplot(x='long_dist_cust_merch', data=data, kde=True)
plt.title('Longitude Distance between Customer and Merchant (Fraud Data)')
plt.xlabel('Longitude Distance')
plt.ylabel('Count')


# Encoding
df_copy = df.copy()

# Label Encoding
le = LabelEncoder()

df_copy

# Select a subset of columns from the copied dataframe and store them in a new dataframe called 'catagories'
catagories = df_copy[['category','gender','state','job','day_name','first','last','merchant']]

# Apply label encoding to each selected column and store the encoded values in the corresponding columns of the copied dataframe
for i in catagories.columns:
    df_copy[i] = le.fit_transform(df_copy[i])
    
# Display the updated copied dataframe
df_copy.head()

# Compute chi-squared test between the selected categorical features and the target variable "is_fraud"
score=chi2(df_copy[['category','gender','state','job','day_name','first','last','merchant']],df_copy.is_fraud)
print('Score is: ',score)

# Create a pandas series with the chi-squared statistics and the selected feature names as the index
pd.Series(score[0],catagories.columns)


import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns from the DataFrame
numeric_columns = df.select_dtypes(include='number')

plt.figure(figsize=(15, 10))
sns.heatmap(numeric_columns.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Define X and y variables for the classification problem
X = df_copy.drop('is_fraud', axis=1)
y = df_copy['is_fraud']

# Split the data into training and testing sets using train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the data after splitting
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#balancing the data
smote = SMOTE()

#before smote technique
print("Before SMOTE: ",y_train.value_counts())

# perform oversampling on the training data to address class imbalance
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# print the value counts of the target variable "is_fraud" after SMOTE
print("After SMOTE: ",y_train_smote.value_counts())

print(X_train_smote.shape)
print(y_train_smote.shape)

# Feature Scaling

scaler = MinMaxScaler()

scaled_X_train = scaler.fit_transform(X_train_smote)
scaled_X_test = scaler.transform(X_test)

#making function to run models

accuracy = []
recall = []


# create a function with name run_model
def run_model(model):
        model.fit(scaled_X_train , y_train_smote) # fit the model on the scaled training data
        y_pred_train = model.predict(scaled_X_train) # get predicted values for the training data
        y_pred_test = model.predict(scaled_X_test) # get predicted values for the test data
        a = accuracy_score(y_test,y_pred_test) # calculate the accuracy score for the test data
        b = recall_score(y_test, y_pred_test) # calculate the recall score for the test data
        
        accuracy.append(a)# append the accuracy score to the accuracy list
        recall.append(b)  # append the recall score to the recall list
        
        # print the evaluation results
        print("Accuracy Score of Train Data: ",accuracy_score(y_train_smote,y_pred_train))
        print("Accuracy Score of Test Data: ",accuracy_score(y_test,y_pred_test))   
        print("Confusion Matrix of Test Data: \n",confusion_matrix(y_test,y_pred_test))
        print("Classification Report: \n",classification_report(y_test,y_pred_test))
        
        
        #Logistic Regression

log_reg = LogisticRegression() # define a logistic regression model
run_model(log_reg) # run the logistic regression model using the run_model function

# generate a heatmap of the confusion matrix for the test data
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, log_reg.predict(scaled_X_test)), annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of Test Data')
plt.xlabel('Predicted')


# print accuracy and recall
print("Accuracy: ",accuracy)
print("Recall: ",recall)

# AUC ROC curve

y_pred_roc = log_reg.predict_proba(scaled_X_test)
y_pred_roc

fpr, tpr, thresholds = roc_curve(y_test, y_pred_roc[:,1])

plt.plot(fpr,tpr)
plt.plot(fpr,fpr,'r-')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.show()

# define a decision tree classifier model
dtc = DecisionTreeClassifier(criterion='gini',max_depth=3 )
run_model(dtc) # run the decision tree classifier model using the run_model function

# decision tree classifier
pd.Series(dtc.feature_importances_,X_train_smote.columns)*100

# Random Forest Classifier
rfc=RandomForestClassifier(n_estimators=100, max_depth=2,criterion='gini')
run_model(rfc)

#  SVM

svc = SVC(kernel='linear',probability=True)
run_model(svc)

svc=SVC(C=1.0,kernel='rbf',gamma='scale')

run_model(svc)

# Fit and predict for each classifier
from sklearn.metrics import accuracy_score

# Logistic Regression
log_reg.fit(scaled_X_train, y_train_smote)
y_pred_log_reg = log_reg.predict(scaled_X_test)

# Decision Tree Classifier
dtc.fit(scaled_X_train, y_train_smote)
y_pred_dtc = dtc.predict(scaled_X_test)

# Random Forest Classifier
rfc.fit(scaled_X_train, y_train_smote)
y_pred_rfc = rfc.predict(scaled_X_test)

# SVM
svc.fit(scaled_X_train, y_train_smote)
y_pred_svc = svc.predict(scaled_X_test)

# Accuracy score for each classifier
print('Accuracy score for Logistic Regression: ', accuracy_score(y_test, y_pred_log_reg))
print('Accuracy score for Decision Tree Classifier: ', accuracy_score(y_test, y_pred_dtc))
print('Accuracy score for Random Forest Classifier: ', accuracy_score(y_test, y_pred_rfc))
print('Accuracy score for SVM: ', accuracy_score(y_test, y_pred_svc))

algorithm = ['LogisticRegression','DecisionTree','RandomForest','linear_svc', 'rbf_svc']


performance=pd.DataFrame({'Algorithms':algorithm,'Accuracy':accuracy,'recall':recall})

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# define the neural network
def create_nn():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# create a function with the name run_model
def run_model():
    nn_model = create_nn()
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    nn_model.fit(scaled_X_train, y_train_smote, epochs=50, batch_size=32, 
                 validation_data=(scaled_X_test, y_test), callbacks=[early_stop])
    
    # get predicted values for training and test data
    y_pred_train = nn_model.predict(scaled_X_train)
    y_pred_test = nn_model.predict(scaled_X_test)
    
    # calculate evaluation metrics and append to accuracy and recall lists
    a = accuracy_score(y_test, y_pred_test)
    b = recall_score(y_test, y_pred_test)
    accuracy.append(a)
    recall.append(b)

    # print evaluation results
    print("Accuracy Score of Train Data: ", accuracy_score(y_train_smote, y_pred_train))
    print("Accuracy Score of Test Data: ", a)
    print("Confusion Matrix of Test Data: \n", confusion_matrix(y_test, y_pred_test))
    print("Classification Report: \n", classification_report(y_test, y_pred_test))

import statistics


# calculate mean and standard deviation of accuracy and recall
mean_accuracy = sum(accuracy) / len(accuracy)
std_accuracy = statistics.stdev(accuracy)
mean_recall = sum(recall) / len(recall)
std_recall = statistics.stdev(recall)

# print descriptive evaluation statistics
print("Average Accuracy: {:.2%} (+/- {:.2%})".format(mean_accuracy, std_accuracy))
print("Average Recall: {:.2%} (+/- {:.2%})".format(mean_recall, std_recall))


from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)

# get predicted values for training and test data
y_pred_train = gbm.predict(X_train)
y_pred_test = gbm.predict(X_test)

# calculate evaluation metrics and append to accuracy and recall lists
a = accuracy_score(y_test, y_pred_test)
b = recall_score(y_test, y_pred_test)
accuracy.append(a)
recall.append(b)

# print evaluation results
print("Accuracy Score of Train Data: ", accuracy_score(y_train, y_pred_train))
print("Accuracy Score of Test Data: ", a)
print("Confusion Matrix of Test Data: \n", confusion_matrix(y_test, y_pred_test))
print("Classification Report: \n", classification_report(y_test, y_pred_test))

# generate a heatmap of the confusion matrix for the test data
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Test Data')
plt.show()

# Report precision
from sklearn.metrics import precision_score
precision= precision_score(y_test, y_pred_test)

print('Precision: {}'.format(precision))

