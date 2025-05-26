from generate import generate, chat

# The data must be the same as the one used during training
data = {
    'user_input': [
        "What is your name?",
        "What is the weather like today?",
    ],
    'model_output': [
        "My name is chatbot, nice to meet you",
        "It looks like sunny.",
    ]
}

# The values must be the same as the ones used during training
d_model = 128
num_heads = 4
num_layers = 4

chat(data, d_model, num_heads, num_layers)        # 'chatbot.pth' is required to run
