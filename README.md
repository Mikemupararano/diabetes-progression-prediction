# Diabetes Progression Prediction
This notebook aims to build a multiple linear regression model to predict the progression of diabetes in patients. It follows a series of steps to prepare the data, train the model, and evaluate its performance.
Dataset
The dataset (diabetes_dirty.csv) contains various attributes related to patients and a target variable indicating the progression of diabetes. Ensure that the CSV file is in the same directory as the notebook or provide the correct path to the file.

## Requirements
Python 3.x
Jupyter Notebook
pandas
scikit-learn
Install the necessary packages using:

bash
Copy code
pip install pandas scikit-learn
Steps
Import Libraries
The notebook begins by importing the necessary libraries:

pandas for data manipulation.
train_test_split from sklearn.model_selection for splitting the data.
MinMaxScaler and StandardScaler from sklearn.preprocessing for data normalization.
LinearRegression from sklearn.linear_model for building the regression model.
r2_score from sklearn.metrics for evaluating the model.
Read the Dataset
The dataset is read into a pandas DataFrame using:

python
Copy code
df = pd.read_csv('diabetes_dirty.csv')
Differentiate Variables
Independent variables (features) are assigned to X, and the dependent variable (target) is assigned to Y:

python
Copy code
X = df.drop('target', axis=1)  # Replace 'target' with the actual target column name
Y = df['target']
Split the Data
The data is split into training (80%) and test (20%) sets:

python
Copy code
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Scaling/Normalization
The necessity for scaling or normalization is investigated, and both MinMaxScaler and StandardScaler are applied:

python
Copy code
## Using MinMaxScaler
min_max_scaler = MinMaxScaler()
X_train_scaled_minmax = min_max_scaler.fit_transform(X_train)
X_test_scaled_minmax = min_max_scaler.transform(X_test)

## Using StandardScaler
standard_scaler = StandardScaler()
X_train_scaled_standard = standard_scaler.fit_transform(X_train)
X_test_scaled_standard = standard_scaler.transform(X_test)
Train the Model
A multiple linear regression model is trained using the training set:

python
Copy code
model = LinearRegression()
model.fit(X_train_scaled_standard, Y_train)
Model Outputs
The intercept and coefficients of the trained model are printed:

python
Copy code
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
Generate Predictions
Predictions are made for the test set:

python
Copy code
Y_pred = model.predict(X_test_scaled_standard)
Compute R-squared
The R-squared score is calculated to evaluate the model:

python
Copy code
r2 = r2_score(Y_test, Y_pred)
print("R-squared:", r2)
Usage
Ensure you have the necessary packages installed.
Place diabetes_dirty.csv in the same directory as the notebook or update the path in the code.
Open diabetes_regression.ipynb in Jupyter Notebook.
Run the cells sequentially to execute the steps and observe the results.

## Conclusion
This notebook demonstrates the process of building a multiple linear regression model to predict diabetes progression. It includes data preprocessing, model training, and performance evaluation, providing a comprehensive workflow for similar regression tasks.