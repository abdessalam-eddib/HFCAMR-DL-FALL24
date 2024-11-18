import pandas as pd
from tqdm import tqdm
import torch

from transformers import pipeline
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Get the model 
model = "/home/ubuntu/abdes/additional_exp/trained_model_33"

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
model = AutoModelForCausalLM.from_pretrained(model,
                                             torch_dtype=torch.bfloat16,
                                             device_map={"":device_string},
                                             low_cpu_mem_usage = True,
                                             quantization_config = quantization_config)

tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B",
                                          use_fast=True,
                                          add_eos_token=True,
                                          max_seq_len=2600)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"

TRUE_KEYWORDS = {
        "true", 
        "correct", 
        "solved",
        "1",
    }
    
FALSE_KEYWORDS = {
        "0",
        "false", 
        "wrong", 
        "incorrect", 
        "not solved", 
        "not correct",
    }

test_prompt = """You are a highly skilled mathematician tasked with evaluating whether an answer to a mathematical question is correct. To provide an accurate evaluation, follow these steps:

1. Read the question carefully to understand what is being asked.
2. Review the answer provided and determine if it logically follows from the question.
3. Examine the explanation given to know whether it provides sufficient reasoning to support the answer, and whether there are any logical flaws or assumptions that need to be addressed.
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


def format_prompt_examples(examples):
    """
    Format a list of examples into structured prompts for evaluation by the model.
    
    Args:
        examples (dict): A dictionary containing the following keys:
            - 'question': A list of mathematical questions.
            - 'answer': A list of corresponding answers.
            - 'solution': A list of explanations for the answers.
    
    Returns:
        dict: A dictionary containing a single key 'text' with a list of formatted prompts.
    """
    
    # Extract relevant columns from the input examples
    questions = examples["question"]
    answers = examples["answer"]
    explanations = examples["solution"]
    
    formatted_prompts = []  # List to store the formatted prompt strings
    
    # Loop through each example and format the prompt
    for question, answer, explanation in zip(questions, answers, explanations):
        # Constructing the formatted prompt 
        formatted_prompt = test_prompt.format(question, answer, explanation, "")
        formatted_prompts.append(formatted_prompt)
    
    return {"text": formatted_prompts}  # Return the list of formatted prompts


test_set = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")
test_set = test_set.map(format_prompt_examples, batched=True, num_proc=200)
results = []
    
generator = pipeline(
        "text-generation",
        model=model,
        device_map={"":device_string},
        tokenizer=tokenizer,
        batch_size=32,
    )
for out in tqdm(generator(KeyDataset(test_set, "text"), return_full_text=False, max_new_tokens=10)):
    generated_text = out[0]["generated_text"].lower().strip()
            
    # Handle negations first
    if "not solved" in generated_text or "not correct" in generated_text:
            prediction = "False"
    # Then handle positive cases
    elif any(keyword in generated_text for keyword in TRUE_KEYWORDS):
            prediction = "True"
    elif any(keyword in generated_text for keyword in FALSE_KEYWORDS):
            prediction = "False"
    else:
            prediction = out[0]["generated_text"]
                
    results.append(prediction)

results_df = pd.DataFrame({
    'ID': range(len(results)),
    'is_correct': results,
})

# Calculate accuracy
results_df.to_csv(f"results.csv", index=False)
