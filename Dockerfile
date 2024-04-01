FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN pip install accelerate transformers datasets evaluate==0.4.0 huggingface-hub==0.14.1 datasketch==1.5.7 nltk==3.8.1 rouge_score==0.1.2

#Optional/unused
# dpu_utils

# Optional: for experiment tracking
# RUN pip install wandb==0.15.0 tensorboard==2.12.2
