import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv('temperatures.csv')

print("First 5 head lines: ", "\n", data.head())
print("--------------------------------------------------------------------------------------")

print("Columns of dataset","\n",data.columns)
print("--------------------------------------------------------------------------------------")
print("Shape of the dataset:", data.shape)
print("--------------------------------------------------------------------------------------")
print("Discribtion of data : \n ",data['JAN-FEB'].describe())

anual_samr = data['ANNUAL'].describe()
print("\nSummary Stat for Annual Temp.:\n", anual_samr)
print("--------------------------------------------------------------------------------------")
print("Data Info :",data.info())

print("--------------------------------------------------------------------------------------")
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
fdata = data[months]

plt.figure(figsize=(10, 6))
plt.plot(data['YEAR'],fdata)
plt.title('Monthly Temperature Variation')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(data['YEAR'], data['JAN'], color='blue', edgecolor='black')
plt.xlabel("Year")
plt.ylabel("Temperature in January")
plt.title("Distribution of Temperature Values for January Across Different Years")
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(data['YEAR'], data['JAN'], marker='+' ,color='blue')
plt.xlabel("Year")
plt.ylabel("Temperature in January")
plt.title("Trend in Temperature Values for January Over the Years")
plt.show()


data.plot(x='YEAR', y='ANNUAL', kind='scatter', figsize=(10, 6))
plt.xlabel("Year")
plt.ylabel("Annual Temperature")
plt.title("Scatter Plot of Annual Temperatures")
plt.show()


x = data['YEAR'].values.reshape(-1, 1)
y = data['ANNUAL'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

print("Regression intercept : ",regressor.intercept_)
print("Regression coeffivient: ",regressor.coef_)

plt.scatter(x_test, y_test)
plt.plot(x_test,y_pred)
plt.show()
