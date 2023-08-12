# Health_Insurance
Exploring Health Insurance Predictions: From Data Analysis to Personalized Estimates 📊💰 #MachineLearning #HealthcareAnalytics

Certainly! This code is an example of using machine learning to predict health insurance charges based on various features such as age, sex, BMI (Body Mass Index), number of children, smoking status, and region. Let's break down the code step by step:

1. **Import Libraries:** The code begins by importing necessary libraries such as `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, and `warnings`. It also imports specific functions like `mean_squared_error`, `r2_score` from `sklearn.metrics`, and `LabelEncoder` from `sklearn.preprocessing`.

2. **Read Data:** The code reads a CSV file named "Health_insurance.csv" into a Pandas DataFrame (`df`) using the `pd.read_csv` function. It then displays the first few rows of the DataFrame using `df.head()`.

3. **Data Exploration:**
   - `df.info()`: Prints information about the DataFrame including data types and non-null counts.
   - `df.isna().sum()`: Calculates and displays the count of missing values in each column.
   - `df.dtypes`: Prints the data types of each column.
   - `df.count`: This is incorrect. It should be `df.count()` and it would return the count of non-null values in each column.

4. **Data Preprocessing:**
   - Label Encoding: The code uses the `LabelEncoder` to encode categorical variables like 'sex', 'smoker', and 'region' into numerical values.
   - Mapping: It maps categorical values to numerical values for 'sex' (female: 0, male: 1) and 'smoker' (no: 0, yes: 1).

5. **Data Visualization:**
   - The code uses various Seaborn functions to create visualizations, including:
     - `sns.boxplot(x=df['bmi'])`: Box plot of the 'bmi' column.
     - `sns.countplot(data=df, x="children")`: Count plot of the 'children' column.
     - `sns.heatmap(df.corr())`: Correlation heatmap of the DataFrame.
     - `sns.distplot(df["charges"])`: Distribution plot of the 'charges' column.
     - `sns.boxplot(x="children", y="age", data=df)`: Box plot of 'age' grouped by 'children' category.

6. **Data Splitting:** The code splits the data into features (`x`) and target (`y`). Then, it uses `train_test_split` from `sklearn.model_selection` to split the data into training and testing sets.

7. **Model Training:** The code imports the `LinearRegression` model from `sklearn.linear_model`, initializes the model, and fits it to the training data using `model.fit(xtrain, ytrain)`.

8. **Prediction and Evaluation:** The code makes predictions using the trained model (`ypred = model.predict(xtest)`). It calculates the R-squared (coefficient of determination) scores for both the training and testing sets to evaluate the model's performance.
   - `model.score(xtrain, ytrain)`: R-squared score on the training set.
   - `model.score(xtest, ytest)`: R-squared score on the testing set.

9. **User Input and Prediction:**
   - The code gathers user input for features like age, sex, BMI, children, smoking status, and region.
   - It encodes the categorical inputs using the label encoders trained earlier.
   - The prepared input data is used to make a health insurance charge prediction using the trained linear regression model.

Overall, the code demonstrates a complete pipeline from data loading and preprocessing to model training, evaluation, and making predictions.
