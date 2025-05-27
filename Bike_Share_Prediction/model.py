# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

#load the required libraries
from sklearn import preprocessing,metrics,linear_model
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split

# Spliting the X and y Values from the dataset
# X is an Independent Variable
X = bike_df.iloc[:, :13]

# y is a dependent variables
y = bike_df.iloc[:, -1]


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)


# Saving model to disk
pickle.dump(regressor, open('lr_model.pkl','wb'))

