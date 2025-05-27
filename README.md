# Transformer-From-Scratch
Transformer has become the game-changing architecture in natural language processing and even other fields. This project is the implementation of Transformer from scratch in PyTorch based on the paper [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

# Overview
You only need run_train.py and run_chatbot.py to get started after download all .py files and required libraries.
- tokenizer.py - preprocessing of texts to be inputs
- transformer.py - the main transformer architecture
- train.py - training loop and saving the model
- generate.py - generation of texts from the model
- run_train.py - data preparation and start of training
- run_chatbot.py - chat with the model

# run_train.py
```python
from train import train

data = {
    'user_input': [
        "What is your name?",
        "What is the weather like today?",
    ],
    'model_output': [
        "My name is chatbot, nice to meet you.",
        "It looks like sunny.",
    ]
}

d_model = 128       # d_model must be divisible by num_heads
num_heads = 4
num_layers = 4
train_num_epochs = 500
train_save_period = 50

train(data, d_model, num_heads, num_layers, train_num_epochs, train_save_period)
```
```text
Epoch 50, Loss: 0.0112
Model saved at epoch 50 with the best loss: 0.0112
Epoch 100, Loss: 0.0047
Model saved at epoch 100 with the best loss: 0.0047
Epoch 150, Loss: 0.0032
Model saved at epoch 150 with the best loss: 0.0032
```

# run_chatbot.py
```python
from generate import chat

# The data must be the same as the one used during training
data = {
    'user_input': [
        "What is your name?",
        "What is the weather like today?",
    ],
    'model_output': [
        "My name is chatbot, nice to meet you.",
        "It looks like sunny.",
    ]
}

# The values must be the same as the ones used during training
d_model = 128
num_heads = 4
num_layers = 4

chat(data, d_model, num_heads, num_layers)        # 'chatbot.pth' is required to run
```
```text
Enter 'exit' to quit.
You: your name?
Bot: My name is chatbot, nice to meet you.
```
