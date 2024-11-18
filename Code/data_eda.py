import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")
dataset = pd.DataFrame(dataset)

# Basic Dataset Information
print("Dataset Info:")
print(dataset.info())

# Descriptive Statistics for Numerical Data (only 'is_correct' will have numerical data)
print("\nDescriptive Statistics:")
print(dataset.describe())

# Display the first few rows to get an overview
print("\nFirst Few Rows of the Dataset:")
print(dataset.head())

# Check for missing values
print("\nMissing Values in Each Column:")
print(dataset.isnull().sum())

# Class distribution in the 'is_correct' column (binary classification check)
plt.figure(figsize=(6, 4))
sns.countplot(data=dataset, x='is_correct', palette='Set2')
plt.xlabel('Is Correct')
plt.ylabel('Count')
plt.show()

# Length Analysis of Text Fields (question, answer, explanation)
dataset['question_length'] = dataset['question'].apply(lambda x: len(x.split()))
dataset['answer_length'] = dataset['answer'].apply(lambda x: len(x.split()))
dataset['explanation_length'] = dataset['solution'].apply(lambda x: len(x.split()))


plt.figure(figsize=(6, 18))  # Adjusting the figure size for vertical alignment

# Plotting text length distributions
plt.subplot(3, 1, 1)
sns.histplot(dataset['question_length'], kde=True, color='blue', bins=30)
plt.xlabel('Question Length')
plt.title('Question Length Distribution')

plt.subplot(3, 1, 2)
sns.histplot(dataset['answer_length'], kde=True, color='green', bins=30)
plt.xlabel('Answer Length')
plt.title('Answer Length Distribution')

plt.subplot(3, 1, 3)
sns.histplot(dataset['explanation_length'], kde=True, color='red', bins=30)
plt.xlabel('Explanation Length')
plt.title('Explanation Length Distribution')

plt.tight_layout()
plt.show()