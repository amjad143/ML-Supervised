


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

landprice2= pd.read_csv(r'C:\Users\AMZAD\Pandas\codes\DataScienceProjects\Dummy-MLR\landprice2.csv')

#landpric = df.dropna()
dummy = pd.get_dummies(landprice2.City).iloc[:,:2]

dataset1 = pd.concat([landprice2,dummy],axis=1)
df2 = dataset1.drop(['City'],axis=1)

dataset = landprice2.drop(['City'],axis=1)


x_ar1 = dataset.iloc[0:,0].values.reshape(-1,1)
y_pr1 = dataset.iloc[0:,3].values.reshape(-1,1)


#splitting the data for training and testing
train_x,test_x,train_y,test_y = train_test_split(x_ar1,y_pr1,test_size=0.35,random_state=0)

model = LinearRegression()
model.fit(train_x,train_y)


pred_y_tr = model.predict(train_x)
pred_y_ts_ar1 = model.predict(test_x)



r_square_ar1 = r2_score(test_y,pred_y_ts_ar1)
r_square = r2_score(train_y,pred_y_tr)


value = np.array([[89]])
discount1 = model.predict(value)





#Plotting the graph
plt.scatter(train_x,train_y,color = 'red')
plt.plot(train_x,model.predict(train_x),color = 'blue')
plt.title("Area and Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()