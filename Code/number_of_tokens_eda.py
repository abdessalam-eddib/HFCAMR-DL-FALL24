import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

train_dataset = load_from_disk("train_sets/shard_extra")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")



prompt = """You are a highly skilled mathematician tasked with evaluating whether an answer to a mathematical question is correct or no. To provide an accurate evaluation, follow these steps:

1. Read the question carefully to understand what is being asked.
2. Review the answer provided and determine if it logically follows from the question.
3. Examine the explanation given to know whether it provides sufficient reasoning to support the answer, and whether there any logical flaws or assumptions that need to be addressed.
4. Based on your assessment of the question, answer, and explanation, decide if the answer is correct or not.
5. Your Output should exactly be 'True' if the answer is correct, otherwise 'False'.
### Question:
{}

### Answer:
{}

### Explanation:
{}

### Output:
{}
"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN to terminate generation

def format_prompt_examples(examples):
    """
    Format a list of examples into structured prompts for evaluation by the model.
    
    Args:
        examples (dict): A dictionary containing the following keys:
            - 'question': A list of mathematical questions.
            - 'answer': A list of corresponding answers.
            - 'solution': A list of explanations for the answers.
            - 'is_correct': A list indicating whether each answer is correct (True/False).
    
    Returns:
        dict: A dictionary containing a single key 'text' with a list of formatted prompts.
    """
    
    # Extract relevant columns from the input examples
    questions = examples["question"]
    answers = examples["answer"]
    explanations = examples["solution"]
    correctness_labels = examples["is_correct"]
    
    formatted_prompts = []  # List to store the formatted prompt strings
    
    # Loop through each example and format the prompt
    for question, answer, explanation, correctness in zip(questions, answers, explanations, correctness_labels):
        # Constructing the formatted prompt with a chain-of-thought structure
        formatted_prompt = prompt.format(question, answer, explanation, correctness) + EOS_TOKEN
        formatted_prompts.append(formatted_prompt)
    
    return {"text": formatted_prompts}  # Return the list of formatted prompts

# Format the examples in the dataset
train_dataset = train_dataset.map(format_prompt_examples, batched=True)

# Display the formatted dataset
print(train_dataset[0]["text"])

# Plot the distribution of the 'is_correct' labels
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(x='is_correct', data=train_dataset.to_pandas())
plt.title("Distribution of Correctness Labels in the Training Dataset")

# Display the plot  
plt.savefig(f"figures/correctness_distribution.png")

# Calculate the percentage of correct and incorrect answers
correct_count = sum(train_dataset['is_correct'])
total_count = len(train_dataset['is_correct'])
correct_percentage = (correct_count / total_count) * 100
incorrect_percentage = 100 - correct_percentage
print(f"Percentage of Correct Answers: {correct_percentage:.2f}%")
print(f"Percentage of Incorrect Answers: {incorrect_percentage:.2f}%")

# Compute the number of tokens in each prompt
token_counts = [len(tokenizer.tokenize(prompt)) for prompt in train_dataset['text']]
token_counts_df = pd.DataFrame(token_counts, columns=['token_count'])
# Plot the distribution of token counts
plt.figure(figsize=(10, 6))
sns.histplot(token_counts_df, bins=30, kde=True)
plt.title("Distribution of Token Counts in Formatted Prompts")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
# Display the plot
plt.savefig(f"figures/token_count_distribution.png")

# Calculate the maximum, minimum, and average token counts	and 95th percentile
max_token_count = max(token_counts)
min_token_count = min(token_counts)
average_token_count = sum(token_counts) / len(token_counts)
percentile_99 = np.percentile(token_counts, 99)
print(f"Maximum Token Count: {max_token_count}")
print(f"Minimum Token Count: {min_token_count}")
print(f"Average Token Count: {average_token_count:.2f}")
print(f"99th Percentile Token Count: {percentile_99}")
