import seaborn as sea
import pandas as pd
from matplotlib import pyplot as plt

# Load your CSV dataset
csv_path = r"D:\College\SY\Python\heart_attack_prediction_dataset.csv"
csv_read = pd.read_csv(csv_path)

# Print the column names to inspect
#print(csv_read.columns)

# Create a scatter plot
plt.scatter(csv_read['Age'], csv_read['Heart Rate'] , alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Heart Rate')
plt.title('Scatter Plot: Age vs. Heart Rate')
plt.show()

plt.bar(csv_read['Age'], csv_read['Heart Rate'])
plt.xlabel('Age')
plt.ylabel('Heart Rate')
plt.title('Bar Plot: Age vs. Heart Rate')
plt.show()

plt.boxplot([csv_read[csv_read['Sex'] == 'Male']['Heart Rate'],csv_read[csv_read['Sex'] == 'Female']['Heart Rate']],
            labels=['Male','Female'])
plt.xlabel('Sex')
plt.ylabel('Heart Rate')
plt.title('Boxplot of Heart Rate by Sex')
plt.show()

plt.hist(csv_read['Age'])

plt.hist2d(csv_read['Age'], csv_read['Heart Rate'], bins=(30, 30), cmap='Blues')
plt.xlabel('Age')
plt.ylabel('Heart Rate')
plt.title('2D Histogram: Age vs. Heart Rate')
plt.colorbar(label='Frequency')
plt.show()



sea.regplot(x='Age', y='Heart Rate', data=csv_read)
plt.xlabel('Age')
plt.ylabel('Heart Rate')
plt.title('Regression Plot: Age vs. Heart Rate')
plt.show()








