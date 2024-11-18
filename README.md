# Heterogeneous Fact-Checking Agents for Mathematical Reasoning: A Consensus-Based Framework for Solution Assessment  

This repository explores a consensus-based framework for assessing the correctness of mathematical solutions using a combination of weak learners using fine-tuning techniques. Below is an overview of the project structure and its purpose.

---

## **Repository Structure**

### 1. **Code**  

Contains all the codes required for data processing, training, inference, and evaluation:  
- **`train`**: Code for fine-tuning using LoRA and GeLoRA.  
- **`inference`**: Code for performing inference on validation and test sets.  
- **`data_eda`**: Code for exploratory data analysis (EDA) on training and test datasets.  
- **`majority_vote`**: Code to aggregate weak learners’ predictions using majority voting.  
- **`number_of_tokens_eda`**: Code to analyze token distribution and determine the maximum token length for tokenizer settings.  
- **`plot_train_loss_grad_norm`**: Code for plotting training loss and gradient norm across epochs.  
- **`primary_sharding`**: Code for sharding the dataset into 20 subsets while maintaining the original train label distribution.  
- **`balanced_data_sharding`**: Code for creating balanced data shards with a validation set included.  
- **`compute_val_metrics`**: Code to compute validation metrics, including accuracy, balanced accuracy, and F1 score.

---

### 2. **EDA**  

Contains figures generated during exploratory data analysis (EDA) on both the training and test datasets. These visualizations provide insights into data distribution, token lengths, and more.

---

### 3. **Report**  

Contains the project report in PDF format, detailing the framework, methodology, experiments, and results.

---

### 4. **Scripts**  

Executable scripts to launch training and inference pipelines. 

---

### 5. **Training Logs**  
Stores logs generated during the training phase, capturing information about loss, accuracy, gradient norms, and other training details.

---

### 6. **Validation Log and Plot**  

- **Log**: Training log recorded during training for analysis and debugging.  
- **Plot**: Validation and training loss plot across steps for performance monitoring.

---

### 7. **Validation Predictions**  

Contains CSV files with predictions made on the validation set during inference.

---

### 8. **Test Predictions**  

Contains CSV files with predictions made on the test set during inference.

---

### 5. **Training Plots**  

Stores plots of training loss and gradient norm across steps for performance and stability monitoring.


---

## **Usage**

### **Dependencies**  

Before running any scripts, ensure you have the following dependencies installed:  
- Python 3.8+  
- PyTorch  
- Transformers (Hugging Face)  
- PEFT & TRL libraries  
- Additional packages (see `requirements.txt`)  

### **Setup**  
1. Clone the repository:  
   ```bash
   git clone [https://github.com/abdessalam-eddib/HFCAMR-DL-FALL24.git](https://github.com/abdessalam-eddib/HFCAMR-DL-FALL24.git)
   cd HFCAMR-DL-FALL24
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

### **Key Scripts**  
- **Train the model**:  
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  nohup torchrun --nproc_per_node=8 code/train.py  > train.log 2>&1 &
  ```  
- **Run inference**:  
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  nohup torchrun --nproc_per_node=8 code/inference.py  > inference.log 2>&1 &
  ```  
- **Generate majority vote**:  
  ```bash
  python code/majority_vote.py
  ```  

---

## **Methodology**  

### **Problem Statement**  
Fact-checking solutions for mathematical reasoning problems requires robust methods that can geenralize well across unseen mathematical problems. This project develops a consensus-based approach using heterogeneous fact-checking agents, combining fine-tuned weak learnerson different subsets of the training set to assess the correctness of solutions.  

### **Approach**  
1. **Fine-Tuning**: Leveraging low rank adaptation techniques for efficient adaptation of large language models to mathematical reasoning tasks.  
2. **Weak Learners**: Using weak learners to diversify predictions and improve robustness.  
3. **Consensus Framework**: Aggregating predictions via weighted majority voting to improve final assessment accuracy,a nd mitigating the inherent bias in the training data.
4. **Evaluation Metrics**: Assessing performance using metrics such as accuracy, balanced accuracy, and F1 score.  

---

## **Results**  
For a detailed overview of the methodology, experimental setup, and results, refer to the **Report** in the `Report/` folder.  

---

## **License**  
This project is licensed under the MIT License. See `LICENSE` for more details.

---

## **Contact**  
For questions or collaborations, reach out to:  
**Abdessalam Ed-dib**  
Email: ae2842@nyu.edu
GitHub: https://github.com/abdessalam-eddib/

Happy fact-checking! 🚀
