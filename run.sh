# Model Training and Inference Guide
# System requirements: 8x NVIDIA A100 80GB GPUs

# =======================================
# Training
# =======================================
# Start training with a single GPU (scale as needed)
python train.py config/config.py

# =======================================
# Inference
# =======================================
# Before running inference:
# 1. Update the 'load_from' parameter in config/config.py with your trained model path
# 2. Run the extraction script with appropriate GPU resources

# Extract neural representations (single GPU)
python extract_neural_rep.py config/config.py

# Verify the cross model alignment.
python lv_statistics_neural_vs_stimulus.py

# Verify the zero-shot preserved neural presentation.
python lv_statistics_activtiy.py
