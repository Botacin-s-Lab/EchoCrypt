from huggingface_hub import login
login(token='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')

import re
import torch
import difflib
import pandas as pd

from transformers import pipeline
from tqdm import tqdm


def get_messages(sentence, examples):
    messages = [
        {"role": "system", "content": "You are an expert in correcting typos in sentences."},
        {"role": "user", "content": """
Here are examples of sentences with typos; learn from them:

{examples}
Now, please correct this sentence and output only the corrected version with no additional text:

{target_sentence}
        """.format(target_sentence=sentence, examples=examples)},
    ]
    return messages

def get_llm_sentence(sentence, examples):
    messages = get_messages(sentence, examples)
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    llm_sentence = outputs[0]["generated_text"][-1]["content"]
    return llm_sentence

def llm_postprocess(sentence):
    sentence = sentence.lower().strip()
    # remove all non a-z0-9 
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    return sentence

def compute_accuracy_and_wrong_syllables(true_sentence, predicted_sentence):
    # Character-level accuracy using SequenceMatcher
    char_matcher = difflib.SequenceMatcher(None, true_sentence, predicted_sentence)
    accuracy = char_matcher.ratio()
    
    # Word-level wrong syllable count using SequenceMatcher on word lists
    true_words = true_sentence.split()
    predicted_words = predicted_sentence.split()
    word_matcher = difflib.SequenceMatcher(None, true_words, predicted_words)
    
    # Calculate wrong syllables based on insert, delete, and replace operations
    wrong_syllables = sum(1 for tag, _, _, _, _ in word_matcher.get_opcodes() if tag in ('insert', 'delete', 'replace'))
    
    return accuracy, wrong_syllables

NFs = [
    "noise_0.012",
    "noise_0.024",
    "noise_0.06",
]
model_id = "meta-llama/Llama-3.2-1B-Instruct"
output_dir = "llama3_2_1b"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

for nf in NFs:
    df = pd.read_csv(f"results/{nf}.csv")
    examples = ""

    for i in range(2):
        examples += f"\tsentence: {df['Predicted Sentence'][i]}\n"
        examples += f"\tcorrected: {df['True Sentence'][i]}\n\n"

    llm_accs = []
    llm_ws = []
    llm_sen = []
    total=len(df)

    for index, row in tqdm(df.iterrows(), total=total):
        should_print = index % 100 == 0
        predicted_sentence = row['Predicted Sentence']
        true_sentence = row['True Sentence']
        accuracy, wrong_syllables = compute_accuracy_and_wrong_syllables(true_sentence, predicted_sentence)
        if should_print:
            print(f"[LLM Auto] Index: {index} of {total}")
            print("[LLM Auto] CoAtNet", accuracy, wrong_syllables)
        
        llm_sentence = get_llm_sentence(predicted_sentence, examples)
        llm_sentence = llm_postprocess(llm_sentence)
        accuracy, wrong_syllables = compute_accuracy_and_wrong_syllables(true_sentence, llm_sentence)
        if should_print:
            print("[LLM Auto] LLM", accuracy, wrong_syllables)
            print("[LLM Auto] ==========")
        
        llm_sen.append(llm_sentence)
        llm_accs.append(accuracy)
        llm_ws.append(wrong_syllables)

    df['LLM Sentence'] = llm_sen
    df['LLM Accuracy'] = llm_accs
    df['LLM Wrong syllables'] = llm_ws

    # average accuracy
    llm_avg_accuracy = sum(llm_accs) / len(llm_accs)
    # sum of wrong syllables
    llm_sum_wrong_syllables = sum(llm_ws)

    print(f"[LLM Auto] Model: {model_id}")
    print(f"[LLM Auto] NF {nf}")
    print(f"[LLM Auto] LLM Average Accuracy: {llm_avg_accuracy}")
    print(f"[LLM Auto] LLM Sum of Wrong Syllables: {llm_sum_wrong_syllables}")
    print("[LLM Auto] ===")
    
    df.to_csv(f'results/{output_dir}/{nf}.csv', index=False)
