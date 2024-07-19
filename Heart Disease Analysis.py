# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Load the dataset
data_path = r'C:\Users\Harsh_ehn0ysj\Downloads\Heart Disease data.csv'
data = pd.read_csv(data_path)

# Display the first few rows to understand the structure
print("Initial Data:\n", data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Fill missing values with column mean if necessary
data.fillna(data.mean(), inplace=True)

# Convert categorical variables to numerical
data['sex'] = data['sex'].map({1: 'male', 0: 'female'})
data['target'] = data['target'].map({1: 'yes', 0: 'no'})

# Create age groups
bins = [0, 30, 45, 60, 75, 90]
labels = ['0-30', '31-45', '46-60', '61-75', '76-90']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)

# Save the cleaned data
cleaned_data_path = r'C:\Users\Harsh_ehn0ysj\Downloads\cleaned_heart_disease_data.csv'
data.to_csv(cleaned_data_path, index=False)

# Summary statistics
print("Summary Statistics:\n", data.describe())

# Data types and non-null counts
print("Data Info:\n", data.info())

# Set the style of the visualization
sns.set_style('whitegrid')

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Gender distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', data=data, palette='coolwarm')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks([0, 1], ['Female', 'Male'])
plt.show()

# Heart disease by age group
plt.figure(figsize=(12, 6))
sns.countplot(x='age_group', hue='target', data=data, palette='Set2')
plt.title('Heart Disease by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

# Average cholesterol levels by heart disease status
if 'target' in data.columns and 'chol' in data.columns:
    avg_cholesterol_by_hd = data.groupby('target')['chol'].mean()
    print("Average Cholesterol Levels by Heart Disease Status:\n", avg_cholesterol_by_hd)

# Maximum Heart Rate Achieved by Heart Disease Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='thalach', data=data, palette='Set3')
plt.title('Maximum Heart Rate Achieved by Heart Disease Status')
plt.xlabel('Heart Disease')
plt.ylabel('Maximum Heart Rate Achieved')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Chest Pain Type Distribution by Heart Disease Status
plt.figure(figsize=(10, 6))
sns.countplot(x='cp', hue='target', data=data, palette='Set2')
plt.title('Chest Pain Type Distribution by Heart Disease Status')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()

# Resting Blood Pressure Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['trestbps'], kde=True, color='green')
plt.title('Resting Blood Pressure Distribution')
plt.xlabel('Resting Blood Pressure (mm Hg)')
plt.ylabel('Frequency')
plt.show()

# Cholesterol Levels Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['chol'], kde=True, color='purple')
plt.title('Cholesterol Levels Distribution')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()

# Fasting Blood Sugar by Heart Disease Status
plt.figure(figsize=(10, 6))
sns.countplot(x='fbs', hue='target', data=data, palette='Set1')
plt.title('Fasting Blood Sugar by Heart Disease Status')
plt.xlabel('Fasting Blood Sugar > 120 mg/dl')
plt.ylabel('Count')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()

# Interactive Plot: Heart Disease by Age Group and Gender
fig = px.bar(data, x='age_group', y='target', color='sex', barmode='group',
             title='Heart Disease by Age Group and Gender',
             labels={'target': 'Heart Disease Count', 'age_group': 'Age Group', 'sex': 'Gender'},
             category_orders={"sex": ['female', 'male'], 'target': ['no', 'yes']})
fig.update_layout(xaxis_title='Age Group', yaxis_title='Heart Disease Count',
                  xaxis={'categoryorder': 'category ascending'})



# Display the key findings and recommendations
print(f"""
### Key Findings
-- **Age Group Analysis**: The age group 46-60 shows the highest prevalence of heart disease.
- **Gender Analysis**: Males are more affected by heart disease compared to females.
- **Cholesterol Levels**: Higher cholesterol levels are associated with heart disease.
- **Maximum Heart Rate**: Patients with heart disease tend to have a lower maximum heart rate.
- **Chest Pain Types**: Type 0 chest pain is more common in heart disease patients.

### Recommendations
1. **Targeted Health Programs**: Develop health awareness programs focusing on age group 46-60.
2. **Regular Screenings**: Encourage routine health check-ups, especially for high-risk demographics.
3. **Lifestyle Interventions**: Promote healthy lifestyle changes, including diet and exercise.
""")
