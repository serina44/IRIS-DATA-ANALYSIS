import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline 

df = pd.read_csv("/content/iris_csv.csv") 
df.head()
df.info()
df.shape
df.describe()
df.isnull().sum()



df.groupby('class').agg(['mean', 'median'])  
df.groupby('class').agg([np.mean, np.median])
plt.figure(figsize=(8,4)) 
sns.boxplot(x='class',y='sepalwidth',data=df ,palette='YlGnBu')
sns.distplot(a=df['petalwidth'], bins=40, color='b')
plt.title('petal width distribution plot')
sns.countplot(x='class',data=df)
sns.heatmap(df.corr(), linecolor='white', linewidths=1)
axis = plt.axes()

axis.scatter(df.sepallength, df.sepalwidth)

axis.set(xlabel='Sepal_Length (cm)',
   ylabel='Sepal_Width (cm)',
   title='Sepal-Length vs Width');
sns.pairplot(df, hue='class')
figure, ax = plt.subplots(2, 2, figsize=(8,8))

ax[0,0].set_title("sepallength")
ax[0,0].hist(df['sepallength'], bins=8)

ax[0,1].set_title("sepalwidth")
ax[0,1].hist(df['sepalwidth'], bins=6);

ax[1,0].set_title("petallength")
ax[1,0].hist(df['petallength'], bins=5);

ax[1,1].set_title("petalwidth")
ax[1,1].hist(df['petalwidth'], bins=5);
