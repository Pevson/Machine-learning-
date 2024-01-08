# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("04-advertising_dataset.csv")


# Display the first few rows of the DataFrame
print(df.head())

# Step 2: Selecting Features for Simple Linear Regression
# Calculate correlation with the target (Sales)
correlation = df.corr()['Sales'].abs().sort_values(ascending=False)
selected_feature = correlation.index[1]  # Select the feature with the highest correlation (excluding 'Sales')
print(f"Selected feature for Simple Linear Regression: {selected_feature}")

# Step 3: Building Simple Linear Regression Model
X_simple = df[selected_feature].values.reshape(-1, 1)
y_simple = df['Sales'].values
X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_simple_train)

# Make predictions
y_simple_pred = simple_model.predict(X_simple_test)

# Calculate Mean Squared Error and Coefficient of Determination for Simple Linear Regression
mse_simple = mean_squared_error(y_simple_test, y_simple_pred)
r2_simple = r2_score(y_simple_test, y_simple_pred)

print(f"\nSimple Linear Regression Results:")
print(f"Mean Squared Error: {mse_simple}")
print(f"Coefficient of Determination (R^2): {r2_simple}")

# Step 4: Building Multiple Linear Regression Model
X_multiple = df[['TV', 'Radio', 'Newspaper']].values
y_multiple = df['Sales'].values
X_multiple_train, X_multiple_test, y_multiple_train, y_multiple_test = train_test_split(X_multiple, y_multiple, test_size=0.2, random_state=42)

multiple_model = LinearRegression()
multiple_model.fit(X_multiple_train, y_multiple_train)

# Make predictions
y_multiple_pred = multiple_model.predict(X_multiple_test)

# Calculate Mean Squared Error and Coefficient of Determination for Multiple Linear Regression
mse_multiple = mean_squared_error(y_multiple_test, y_multiple_pred)
r2_multiple = r2_score(y_multiple_test, y_multiple_pred)

print(f"\nMultiple Linear Regression Results:")
print(f"Mean Squared Error: {mse_multiple}")
print(f"Coefficient of Determination (R^2): {r2_multiple}")
