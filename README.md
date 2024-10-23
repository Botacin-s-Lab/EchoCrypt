# CSCE689_LLM


## Experiments and Results

### Reference
Optimizer: Adam.
The following are all test accuracies.
Zoom Model: 98%
Phone Model: 97%
Zoom Model on Phone Dataset: 3%
Phone Model on Zoom Dataset: 2%
Both Model: 97%
Both Models on Phone Dataset: 99%
Both Models on Zoom Dataset: 98%

### Baselines 
5 models
we test each raw model 0% 30%
fine-tune
we test fine-tuned model (how much improvement) 


craft sentences --> use CoAtNet 24m --> 70%
1000 senteces. --> randomly concate keystrokes --> sentence wav
sentence wav --> isolator --> char wav --> CoAtNet --> recover char. e.g. good morning (goad morning)  diff 1

craft sentences --> use CoAtNet + LLM --> 90%
