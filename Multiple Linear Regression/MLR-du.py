


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

landprice2= pd.read_csv(r'C:\Users\AMZAD\Pandas\codes\DataScienceProjects\Dummy-MLR\landprice2.csv')

#landpric = df.dropna()

dataset1 = pd.concat([landprice2,dummy],axis=1)
dummy = pd.get_dummies(landprice2.City).iloc[:,:2]

df2 = dataset1.drop(['City'],axis=1)
original = df2.sort_index(axis=1,ascending=True)

dataset = landprice2.drop(['City'],axis=1)


x_all = original.iloc[:,:5].values
y_pr1 = original.iloc[0:,5].values


#splitting the data for training and testing
train_x,test_x,train_y,test_y = train_test_split(x_all,y_pr1,test_size=0.30,random_state=0)

model = LinearRegression()
model.fit(train_x,train_y)


pred_all_tr = model.predict(train_x)
pred_all_ts_ar1 = model.predict(test_x)



r_square_all = r2_score(test_y,pred_all_ts_ar1)
r_square_all_tr = r2_score(train_y,pred_y_tr)


value = np.array([[89]])
discount1 = model.predict(value)





#Plotting the graph
plt.scatter(train_x,train_y,color = 'red')
plt.plot(train_x,model.predict(train_x),color = 'blue')
plt.title("Area and Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()