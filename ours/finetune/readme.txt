The first step is to use the "format_convos.py" for formatting conversations in a csv file to a dataset arrow file so
that we can train and test on it.
Replace line 5 with the input csv file (which needs to have True Sentence and Predicted Sentence where the
True Sentence is the ground truth and the Predicted Sentence is with typos). Replace line 27 for the output file name.
The output will be a folder, only the .arrow file within the folder is needed for training and testing.

Place the arrow files in the same directory as the ipynb file.


To load an LoRA model, you can load from scratch or the pretrained ones done by us before. To differentiate, the blocks
to run in the notebook are named as the following respectively:
Training from scratch (LoRA).
and
Training from pretrained LoRA, by Yichen

Within the pretrained LoRA by us, you probably need to load it from online by running:
"Load pretrained from uploaded weights"
You can also run from local pretrained weights, as indicated in the notebook - when you have got a newer ft weights.

Next, you prepare the data for processing. Run the blocks one by one. The print statements help you peek into the data
set so that you know it is getting the correct sentences. Note that you can comment out and only load "dataset" (for
training) or "testset" (for testing).

Then you may train the model by running the blocks one by one.

The Inference section is for running inference on the finetuned model, it can be run directly after training or after
loading a trained model and data prep. The script will create a csv file called llm_res_.csv which will contain
True Sentence, Predicted Sentence,LLM Sentence. The LLM Sentence is the corrected sentence by LLM.

Lastly, you may save the model. The saved model is a folder so there is a script for zipping it.

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