import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load dataset
df = pd.read_csv("05-insurance_charges.csv")

#Explore dataset
print(df.info())
print(df.describe())
print(df.head())

#Visualize the distribution of the target variable (charges)
plt.figure(figsize=(10, 6))
plt.hist(df['charges'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of Insurance Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

#Encode categorical variables (sex, smoker, region)
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

#Select features (X) and target variable (y)
X = df_encoded.drop('charges', axis=1)
y = df['charges']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Build and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Coefficient of Determination (R^2): {r2}")
