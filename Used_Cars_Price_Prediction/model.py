# Importing Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle
import warnings; warnings.filterwarnings('ignore')


# Loading the Dataset
data = pd.read_csv("cars_24_combined.csv")
df = data.drop(columns = 'Unnamed: 0')
df = df.dropna()

df['Year'] = df['Year'].astype(int)
# as number of owner doesn't haave major impact we will drop this column
df =  df.drop(columns = 'Owner')
# As per location we will dtop this column too.
df =  df.drop(columns = 'Location')


# Splitting the dataset in X and y
X = df.drop(columns = 'Price')
y = df['Price']

# Spliting the Data into the X_train, y_train, X_test, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7302)

# One hot Encoding
ohe = OneHotEncoder()
ohe.fit(X[['Car Name','Fuel','Drive','Type']])

column_trans = make_column_transformer((OneHotEncoder(categories = ohe.categories_),['Car Name','Fuel','Drive','Type']),remainder = 'passthrough')


# MKING  lINEAR rEGRESSION
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(X_train, y_train)


# Saving model to disk
pickle.dump(pipe, open('carprice_model.pkl','wb'))

                    









                                       


