#!/bin/bash

# Example script to run the test_zoom.py script
# Make sure you have a trained model in the specified seed folder

# Example 1: Test a SWIN model from "test_run" folder on noisy audio dataset
python test_zoom.py \
    --model swin \
    --seed_folder test_run \
    --test_dataset ../../img_noise_audio_dataset_zoom \
    --batch_size 16 \
    --output_dir test_results_swin

# Example 2: Test a SWIN model from "st_seed0" folder on noisy audio dataset
# python test_zoom.py \
#     --model swin \
#     --seed_folder st_seed0 \
#     --test_dataset ../../img_noise_audio_dataset_zoom \
#     --batch_size 32 \
#     --output_dir test_results_swin

# Example 3: Test a CLIP model from custom seed folder on different dataset
# python test_zoom.py \
#     --model clip \
#     --seed_folder my_custom_seed \
#     --test_dataset ../../img_dataset_zoom \
#     --batch_size 8 \
#     --output_dir test_results_clip

echo "Test completed! Check the output directory for results."
