#!/usr/bin/env python
# coding: utf-8

# In[298]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[299]:


#read csv file
df=pd.read_csv("Desktop\\csvdemo.csv")
df


# In[316]:


#fetch values in array form
point_x=df["Head size"].values
point_y=df["Brain weight"].values
print("x =",point_x,"Y =",point_y)


# In[327]:


#calculating b0,b1 or m and c value for equation y=b1x+b0 or y=mx+c
mean_x=np.mean(point_x) #x bar
mean_y=np.mean(point_y) #y bar
n=len(point_x)
ab=c=0
for val in range(n):   
    ab+=(point_x[val]-mean_x)*(point_y[val]-mean_y)
    c+=(point_x[val]-mean_x)**2
b1=ab/c
b0=mean_y-(b1*mean_x)

print(b1,b0)


# In[326]:


#Plotting values and regression line
maxx=np.max(point_x)
minx=np.min(point_x)

x=np.linspace(minx,maxx)
y=(b1*x)+b0
print(y)

plt.plot(x,y,color="red",label="Regression Line")
plt.scatter(point_x,point_y,label="Scatter Plot")
plt.xlabel("Head size")
plt.ylabel("Brain weight")
plt.legend()
plt.show()


# In[341]:


#calculating root_square problem
up=ny=0
for i in range(n):
    ypred=(b1*point_x[i])+b0
    up+=(ypred-mean_y)**2
    ny+=(point_y[i]-mean_y)**2
ans=up/ny
ans


#using sklearn lib

point_x=point_x.reshape((n,1))
reg=LinearRegression()
reg=reg.fit(point_x,point_y)
y_pred=reg.predict(point_x)
r2=reg.score(point_x,point_y)
print(r2)
    

