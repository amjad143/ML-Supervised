


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

baseball = pd.read_csv(r'C:\Users\AMZAD\Pandas\codes\DataScienceProjects\PolynomialRegression\baseballplayer.csv')


x = baseball.iloc[0:,0].values
y = baseball.iloc[0:,1].values


x1 = np.reshape(x,(-1,1))
y1 = np.reshape(y,(-1,1))

#LinearRegression
model = LinearRegression()
model.fit(x1,y1)

#PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
tran_x = poly_reg.fit_transform(x1)

#Applying PolynomialFeature to LinearRegression
regressor = LinearRegression()
regressor.fit(tran_x,y1)


#predecting LinearRegression values and R2
pred_y = model.predict(x1)
r2 = r2_score(y1,pred_y)


#Predecting R^2 for the PolynomialFeatures
pred_y_pol = regressor.predict(tran_x)
r2_pol = r2_score(y1,pred_y_pol)

# Creating the Templete for the PolynomialFeaures

# Visualising the data

plt.scatter(x1,y1,color = 'red')
plt.plot(x1,regressor.predict(tran_x),color = 'blue')
plt.title("Angel and Distance")
plt.xlabel("Angel")
plt.ylabel("Distance")
plt.show()