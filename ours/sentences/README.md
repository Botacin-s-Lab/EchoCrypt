# Sentences Dataset Subdirectory
This subdirectory contains files and data related to the **EnglishTense** dataset. The dataset is used for training and evaluating language models. It consists of 13,316 sentences and is divided into a finetuning set and an evaluation set.

## Dataset Citation
Please cite the dataset as follows:
```
Ayman, Umme Ayman; Rahman, Md. Hafizur; Islam, Md. Shafiqul (2024), “EnglishTense: A large scale English texts dataset categorized into three categories: Past, Present, Future tenses.”, Mendeley Data, V1, doi: 10.17632/jnb2xp9m4r.1
```

## Dataset Overview
- **Total Sentences**: 13,316
  - **Evaluation**: 1,000 sentences for evaluation
  - **Finetuning**: Remaining sentences for finetuning
- **Sentence Format**: 
  - No punctuation (e.g., "end-to-end" is written as "endtoend").
  - The first 941 sentences contain digits.

## Files in this Directory
This subdirectory contains the following files:

### 1. `EnglishTenseUniqueDataset.xlsx`
The original dataset in Excel format. It contains all the sentences categorized by their tenses (Past, Present, Future).

### 2. `finetune_ds.csv`
A CSV file containing sentences that have passed through the model. It includes the following columns:
- **True Sentence**: The original sentence.
- **Predicted Sentence**: The sentence after being processed by the model.
- **Levenshtein Distance**: The edit distance between the original and predicted sentence.
- **Model**: The model used for generating the prediction.
- **Noise Factor**: The level of noise used during the inference process.

### 3. `ft_sentences.txt`
A plain text file containing the sentences that are used for finetuning the model.

### 4. `main.ipynb`
A Jupyter Notebook to generate the dataset, process it, and apply it for finetuning purposes.

### 5. `README.md`
This README file, which explains the structure and contents of the subdirectory.

### 6. `sentences.txt`
A plain text file containing sentences that are used for evaluation. These sentences are kept separate from the finetuning data to evaluate the model’s performance.

## Directory Structure

```
/home/ali/EchoCrypt/ours/sentences
├── EnglishTenseUniqueDataset.xlsx  # original dataset
├── finetune_ds.csv                 # passed through the model
├── ft_sentences.txt                # separated sentences for finetuning
├── main.ipynb                      # Jupyter notebook to generate the dataset
├── README.md                       # this file
└── sentences.txt                   # sentences used for evaluation
```

## Usage
You don't have to run any code, the data is ready-to-use. You can then use the `finetune_ds.csv` file for training your models and evaluate their performance on the sentences found in `sentences.txt`.
