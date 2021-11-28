

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



df = pd.read_csv(r'C:\Users\AMZAD\Pandas\codes\DataScienceProjects\MLR-2\landpric.csv')
landpric = df.dropna()




x = landpric.iloc[0:,0:3].values
y = landpric.iloc[0:,3].values


train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.40,random_state=0)



model = LinearRegression()
model.fit(train_x,train_y)

plt.scater(x,y,colr = 'red')
plt.title('Area and Price')
plt.xlabel('Area')
plt.ylabel('price')
plt.show()
