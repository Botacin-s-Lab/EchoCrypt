import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Load the CSV file
csv_file = "<repo_root_path>/ours/zoom/results/ft_zoom_ds.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_file)


# Ensure the CSV has the required columns
if not {'True Sentence', 'Predicted Sentence'}.issubset(data.columns):
    raise ValueError("CSV file must contain 'original_sentence' and 'sentence_with_typos' columns")

def get_messages(sentence, examples):
    messages = [
        {"from": "system", "value": "You are an expert in correcting typos in sentences."},
        {"from": "human", "value": """
Here are examples of sentences with typos; learn from them:

{examples}
Now, please correct this sentence and output only the corrected version with no additional text:

{target_sentence}
        """.format(target_sentence=sentence, examples=examples)},
    ]
    return messages

# for phone
example_map = {
    '0.012': """    sentence: by 2480 geneticlengineers will have createf organisms ca9able of surviving inbspace withoutflife suiiort
    corrected: by 2480 genetic engineers will have created organisms capable of surviving in space without life support

    sentence: vy 2510 the local government will have been investing in freen infrastructure for generations
    corrected: by 2510 the local government will have been investing in green infrastructure for generations""",
    '0.024': """    sentence: hy 248u g5netic enh8neers 3ikk hav5 created organisms ca8ableaof survivin6 in siac5 3ithour life suiiort
    corrected: by 2480 genetic engineers will have created organisms capable of surviving in space without life support

    sentence: by 151k the kocak government sikk have7been 8nvestinh in green infrastructure for henerarions
    corrected: by 2510 the local government will have been investing in green infrastructure for generations""",
    '0.06': """    sentence: bh 1489 genetic engin555s 6ikk havexc558tedtokg8ni3m3 caj8hk5 if su5vivinh in so8ce 3ithout kif6 suiji5t
    corrected: by 2480 genetic engineers will have created organisms capable of surviving in space without life support

    sentence: bh 1510 th5 kocak 6ove5nm6nt 3ikk have been inv5sting in g5een inftakt58ct8f6 fo5 g6n558tiink
    corrected: by 2510 the local government will have been investing in green infrastructure for generations"""
}

# for zoom
example_map = {
    '0.1': """    sentence: in the future sustainable tfanspoftation options will st8ll be sought to ease congest8on
    corrected: in the future sustainable transportation options will still be sought to ease congestion

    sentence: edycational divefsity 8s a hallmafk of fofeign academic institutions
    corrected: educational diversity is a hallmark of foreign academic institutions""",
    '0.5': """    sentence: 8b the 5utufe sustaiba8pe t5ans8o5tatipn pptipns will st8llxbe soughtktp eaae cpngestion
    corrected: in the future sustainable transportation options will still be sought to ease congestion

    sentence: xhina haa been actively 8nvopved in peaxekee8ing misaions and humabitafiab ef5pfta
    corrected: china has been actively involved in peacekeeping missions and humanitarian efforts""",
    '1.0': """    sentence: in the 5ut7ee wuqtainabqegteanqpo5tat88n 8p5iona wikp a5ill be a8ught 5o eaae cpngeation
    corrected: in the future sustainable transportation options will still be sought to ease congestion

    sentence: china haa been aativelu involaee in peacekeeping miaa8onw qbe h7man85arian eato55s
    corrected: china has been actively involved in peacekeeping missions and humanitarian efforts"""
}

# Create the conversations
arrow_data_dict = {"conversations": []}
for _, row in tqdm(data.iterrows(), total=len(data)):
    nf = row['Noise Factor']
    examples = example_map[str(nf)]
    conversation = get_messages(row['Predicted Sentence'], examples)
    gpt_response = {"from": "gpt", "value": row['True Sentence']}
    conversation.append(gpt_response)
    arrow_data_dict["conversations"].append(conversation)

# Convert to a Hugging Face Dataset
dataset = Dataset.from_dict(arrow_data_dict)

# Save the dataset as an Arrow file
arrow_file = "zoom_ds"  # Replace with your desired Arrow file name
dataset.save_to_disk(arrow_file)

print(f"Dataset saved to {arrow_file}")
