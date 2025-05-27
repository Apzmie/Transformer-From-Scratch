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
