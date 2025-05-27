# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

#import the csv file
bike_df=pd.read_csv("hour.csv")

#Rename the columns
bike_df.rename(columns={'instant':'rec_id','dteday':'datetime','yr':'year','mnth':'month','weathersit':'weather_condition',
                       'hum':'humidity','cnt':'total_count'},inplace=True)

# Changing the dtype of  necessary columns to categorical columns
bike_df['season']=bike_df.season.astype('category')
bike_df['year']=bike_df.year.astype('category')
bike_df['month']=bike_df.month.astype('category')
bike_df['holiday']=bike_df.holiday.astype('category')
bike_df['weekday']=bike_df.weekday.astype('category')
bike_df['workingday']=bike_df.workingday.astype('category')
bike_df['weather_condition']=bike_df.weather_condition.astype('category')

# Dropping Unnecessary Columns
bike_df.drop('datetime',axis=1,inplace=True)
bike_df.drop('rec_id',axis=1,inplace=True)
bike_df.drop('year',axis=1,inplace=True)


# Spliting the X and y Values from the dataset
# X is an Independent Variable
X = bike_df.iloc[:, :13]

# y is a dependent variables
y = bike_df.iloc[:, -1]

# Splitting Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# Decision tree
from sklearn.tree import DecisionTreeRegressor
DecisionregModel = DecisionTreeRegressor(max_depth=8)
DecisionregModel.fit(X_train,y_train)


# Saving model to disk
pickle.dump(DecisionregModel, open('dt_model.pkl','wb'))

