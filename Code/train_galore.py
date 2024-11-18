import torch
import pandas as pd

from accelerate import PartialState
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import (
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
    SFTConfig,
)

# Pick the dataset shard to train on
dataset_shard = "train_sets/last_shard"

# Set the maximum sequence length
max_seq_length = 2500

# Setting the device
device_string = PartialState().process_index

# Quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-8B",
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation = "flash_attention_2", 
                                             device_map={"":device_string},
                                             low_cpu_mem_usage = True,
                                             use_cache = False,)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B",
                                          use_fast=True,
                                          add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"

# Format the prompt for the model
prompt = """
You are a highly skilled mathematician tasked with evaluating whether an answer to a mathematical question is correct. Follow these steps to provide an accurate evaluation:

1. Read the question carefully to understand the problem being asked.
2. Review the provided answer to determine if it logically follows from the question and correctly solves the problem.
3. Examine the explanation to check if it provides sufficient reasoning to support the answer, addressing any logical flaws or assumptions.
4. Based on your assessment of the question, answer, and explanation, decide if the answer is correct.
5. Output exactly 'True' if the answer is correct, otherwise 'False'.

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
        # Constructing the formatted prompt 
        formatted_prompt = prompt.format(question, answer, explanation, correctness) + EOS_TOKEN
        formatted_prompts.append(formatted_prompt)
    
    return {"text": formatted_prompts}  # Return the list of formatted prompts

# Load the dataset
dataset = load_from_disk(dataset_shard)
train_set = dataset.map(format_prompt_examples, batched=True, num_proc=200, remove_columns=["solution", "is_correct", "answer", "question"])

# Define the data collator, we compute loss on the output only
response_template = "### Output:\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


# Define the training arguments
training_arguments = SFTConfig(
    output_dir=f"./results",
    logging_dir=f"./logs",
    # Increased batch size for better throughput, assuming sufficient GPU memory
    per_device_train_batch_size=2,
    # Increased gradient accumulation for effective larger batch size
    gradient_accumulation_steps=16,
    # Standard learning rate for transformer models
    learning_rate=5e-5,
    # Increased epochs for better convergence
    num_train_epochs=3,
    gradient_checkpointing=True,
    # Standard clipping value for stability
    max_grad_norm=0.3,
    # Reduced weight decay to prevent over-regularization
    weight_decay=0.01,
    # Standard warmup ratio
    warmup_ratio=0.1,
    optim="galore_adamw_8bit",
    optim_target_modules=["attn", "mlp"],
    lr_scheduler_type="constant",
    optim_args = f"rank={1024}, update_proj_gap={50}, scale={2}",
    group_by_length=True,
    report_to="none",
    # Changed to epoch-based saving
    save_strategy="steps",
    save_steps=1,
    # More reasonable logging frequency
    logging_strategy="steps",
    logging_steps=1,
    # Removed max_steps to use num_train_epochs instead
    save_total_limit=1,
    ddp_find_unused_parameters=False,
    disable_tqdm=False,
    max_seq_length = max_seq_length,
    dataset_text_field = "text",
    dataset_num_proc = 200,
)

# Define the Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_set,
    args = training_arguments,
    data_collator = collator,
)

# Train the model
print("Training the model...")
train = trainer.train()

# Save the log history
log_history = pd.DataFrame(trainer.state.log_history)
log_history.to_csv(f"./log.csv", index=False)

# Save the model
trainer.save_model(f"./trained_model")

print("Training completed successfully!")

