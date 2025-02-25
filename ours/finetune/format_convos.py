import pandas as pd
from datasets import Dataset, DatasetDict

# Load the CSV file
csv_file = "2.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_file)


# Ensure the CSV has the required columns
if not {'True Sentence', 'Predicted Sentence'}.issubset(data.columns):
    raise ValueError("CSV file must contain 'original_sentence' and 'sentence_with_typos' columns")

# Create the conversations
arrow_data_dict = {"conversations": []}
for _, row in data.iterrows():
    conversation = [
        {"from": "system", "value": "You are an expert in correcting typos and mistakes in sentences."},
        {"from": "human", "value": f"Correct typos the in following sentence and return only the sentence: {row['Predicted Sentence']}"},
        {"from": "gpt", "value": row['True Sentence']},
    ]
    arrow_data_dict["conversations"].append(conversation)

# Convert to a Hugging Face Dataset
dataset = Dataset.from_dict(arrow_data_dict)

# Save the dataset as an Arrow file
arrow_file = "phone_2_test.arrow"  # Replace with your desired Arrow file name
dataset.save_to_disk(arrow_file)

print(f"Dataset saved to {arrow_file}")
