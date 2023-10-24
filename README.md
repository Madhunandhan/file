import pandas as pd

# Replace the file path with the correct path to your CSV file
file_path = "C:/Users/madhu/downloads/covid_dataset 1.csv"

# Specify data types for columns with mixed types
column_data_types = {
    'DIABETES': float, 'ASTHMA': float, 'HYPERTENSION': float, 'CARDIOVASCULAR': float,
    'OBESITY': float, 'RENAL_CHRONIC': float, 'TOBACCO': float,
    # ... Specify other column data types here if needed
}

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(file_path, dtype=column_data_types, low_memory=False)

# Replace NaN values with 0 in the entire DataFrame
data.fillna(0, inplace=True)

# Display the first few rows of the DataFrame with replaced values
print(data.head())
data.columns
# this function shows all the names of the columns
# Display the last few rows of the dataframe
# Replace NaN values with 0
data.fillna(0, inplace=True)
print(data.tail())
data.describe()
data.nunique()
# this function counts the distinct observation over the the requested axis
data.isnull()
data.tail()
# shows last 10 columns 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Replace the file path with the correct path to your CSV file
file_path = "C:/Users/madhu/downloads/covid_dataset 1.csv"

# Specify data types for columns with mixed types
column_data_types = {
    'DIABETES': float, 'ASTHMA': float, 'HYPERTENSION': float, 'CARDIOVASCULAR': float,
    'OBESITY': float, 'RENAL_CHRONIC': float, 'TOBACCO': float,
    # ... Specify other column data types here if needed
}

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(file_path, dtype=column_data_types, low_memory=False)

# Replace NaN values with 0 in the entire DataFrame
data.fillna(0, inplace=True)

# Display the first few rows of the DataFrame with replaced values
print(data.head())
# Bar Plot: Counts of Gender
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='SEX')
plt.title('Counts of Gender')
plt.show()
# Pie Chart: Distribution of Stages of COVID Disease
plt.figure(figsize=(12, 8))
data['STAGES OF COVID DISEASE'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Distribution of Stages of COVID Disease')
plt.show()
# Histogram: Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='AGE', bins=20, kde=True)
plt.title('Age Distribution')
plt.show()
# Box Plot: Age Distribution by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='SEX', y='AGE')
plt.title('Age Distribution by Gender')
plt.show()
# Scatter Plot: Age vs. Pneumonia
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='AGE', y='PNEUMONIA')
plt.title('Age vs. Pneumonia')
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create DataFrame
data = pd.DataFrame({
    'AGE': [45, 32, 28, 50, 62, 38, 55, 29, 41, 48],
    'HIPERTENSION': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'OBESITY': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    'TOBACCO': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
})

# Create the pair plot
sns.pairplot(data[['AGE', 'HIPERTENSION', 'OBESITY', 'TOBACCO']])
plt.title('Pairwise Relationships')
plt.show()

import matplotlib.pyplot as plt

data = {
    'DIABETES': [10, 5, 2],
    'COPD': [3, 7, 1],
    'ASTHMA': [8, 2, 3],
    'INMSUPR': [1, 0, 2],
    'HIPERTENSION': [15, 8, 10],
    'OTHER_DISEASE': [5, 3, 4],
    'CARDIOVASCULAR': [3, 1, 0],
    'OBESITY': [6, 2, 5],
    'RENAL_CHRONIC': [2, 0, 1],
    'TOBACCO': [8, 4, 6],
    'PATIENT_TYPE': ['A', 'B', 'C']
}

# Create a bar plot for each category
plt.figure(figsize=(12, 8))
for category, values in data.items():
    if category != 'PATIENT_TYPE':
        plt.bar(data['PATIENT_TYPE'], values, label=category)

plt.xlabel('Patient Type')
plt.ylabel('Count')
plt.title('Distribution of Medical Conditions by Patient Type')
plt.legend()
plt.show()
import matplotlib.pyplot as plt

# Medical conditions and their corresponding columns
medical_conditions = ['DIABETES', 'ASTHMA', 'HYPERTENSION', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']


# Calculate the counts for each medical condition
condition_counts = [sum(data[condition]) for condition in medical_conditions]

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(medical_conditions, condition_counts)
plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.title('Counts of Medical Conditions')
plt.xticks(rotation=45)
plt.show()
import matplotlib.pyplot as plt

# Medical conditions and their corresponding columns
medical_conditions = ['DIABETES', 'ASTHMA', 'HYPERTENSION', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']


# Calculate the counts for each medical condition
condition_counts = [sum(data[condition]) for condition in medical_conditions]

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(medical_conditions, condition_counts)
plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.title('Counts of Medical Conditions')
plt.xticks(rotation=45)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
data["DATE_DIED"] = pd.to_datetime(data["DATE_DIED"], errors='coerce')

covid_cases_over_time = data.groupby("DATE_DIED").size()

plt.figure(figsize=(10, 6))
covid_cases_over_time.plot()
plt.title("COVID Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Convert 'LEVEL OF VENTILATOR INTUBATION' column to strings
data['LEVEL OF VENTILATOR INTUBATION'] = data['LEVEL OF VENTILATOR INTUBATION'].astype(str)

# Check if 'AGE' column contains numeric values
data['AGE'] = pd.to_numeric(data['AGE'], errors='coerce')

# Pairwise Scatter Plot: Age vs. Level of Ventilator Intubation
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='AGE', y='LEVEL OF VENTILATOR INTUBATION')
plt.title('Age vs. Level of Ventilator Intubation')
plt.xlabel('Age')
plt.ylabel('Level of Ventilator Intubation')
plt.show()
print(data.columns)
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
import time

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path, low_memory=False)
# Medical conditions and their corresponding columns
medical_conditions = ['DIABETES', 'ASTHMA', 'HYPERTENSION', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']

# Calculate the counts for each medical condition
condition_counts = [sum(data[condition]) for condition in medical_conditions]

# Create scatter plot for medical condition counts
import matplotlib.pyplot as plt
plt.scatter(medical_conditions, condition_counts, marker='o')
plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.title('Counts of Medical Conditions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Convert 'LEVEL OF VENTILATOR INTUBATION' column to strings
data['LEVEL OF VENTILATOR INTUBATION'] = data['LEVEL OF VENTILATOR INTUBATION'].astype(str)

# Check if 'AGE' column contains numeric values
data['AGE'] = pd.to_numeric(data['AGE'], errors='coerce')

# Pairwise Scatter Plot: Age vs. Level of Ventilator Intubation
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='AGE', y='LEVEL OF VENTILATOR INTUBATION')
plt.title('Age vs. Level of Ventilator Intubation')
plt.xlabel('Age')
plt.ylabel('Level of Ventilator Intubation')
plt.show()

# Display DataFrame columns
print(data.columns)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Replace the file path with the correct path to your CSV file
file_path = "C:/Users/madhu/downloads/covid_dataset 1.csv"

# Read the CSV file into a Pandas DataFrame
column_data_types = {
    'DIABETES': float, 'ASTHMA': float, 'HYPERTENSION': float, 'CARDIOVASCULAR': float,
    'OBESITY': float, 'RENAL_CHRONIC': float, 'TOBACCO': float,
}

data = pd.read_csv(file_path, dtype=column_data_types, low_memory=False)
# Replace NaN values with 0 in the entire DataFrame
data.fillna(0, inplace=True)
# Display the first few rows of the DataFrame with replaced values
print(data.head())

# Medical conditions and their corresponding columns
medical_conditions = ['DIABETES', 'ASTHMA', 'HYPERTENSION', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']

# Features: Medical condition columns
X = data[medical_conditions]

# Target: ICU column
y = data['ICU']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
model = RandomForestClassifier(random_state=42)

# Apply k-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Fine-tune hyperparameters using Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Make predictions with the best model from grid search
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
