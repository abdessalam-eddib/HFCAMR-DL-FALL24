import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Load the CSV file
df = pd.read_csv('majority_vote_trial_diversity.csv')

# Turn values to string
df['is_correct'] = df['is_correct'].astype(str)
df['is_correct_truth'] = df['is_correct_truth'].astype(str)

# Turn values different from 'True' to 'False'
df['is_correct'] = df['is_correct'].apply(lambda x: 'False' if x != 'True' else x)

# Turn to boolean
df['is_correct'] = df['is_correct'].apply(lambda x: x == 'True')
df['is_correct_truth'] = df['is_correct_truth'].apply(lambda x: x == 'True')

print(df["is_correct"].value_counts())

# Assuming 'is_correct' is the predicted label and 'is_correct_truth' is the true label
y_pred = df['is_correct']
y_true = df['is_correct_truth']

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate balanced accuracy
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

# Calculate F1 score (you can choose the average method, here we use 'binary' for binary classification)
f1 = f1_score(y_true, y_pred, average='binary')

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")