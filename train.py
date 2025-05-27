from tokenizer import tokenizer
from transformer import Transformer
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def train(data, d_model, num_heads, num_layers, train_num_epochs, train_save_period):
    df = pd.DataFrame(data)
    user_input = df['user_input'].tolist()
    model_output = df['model_output'].tolist()
    enc_input_tensor, dec_input_tensor, dec_target_tensor, vocab_size, enc_max_len, dec_max_len, vocab = tokenizer(user_input, model_output)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc_input_tensor = enc_input_tensor.to(device)
    dec_input_tensor = dec_input_tensor.to(device)
    dec_target_tensor = dec_target_tensor.to(device)

    model = Transformer(d_model, num_heads, num_layers, vocab_size, enc_max_len, dec_max_len)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    optimizer = optim.AdamW(model.parameters())

    best_loss = float('inf')
    for epoch in range(train_num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(enc_input_tensor, dec_input_tensor, vocab)
        target = dec_target_tensor
        loss = loss_fn(output.reshape(-1, vocab_size), target.reshape(-1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % train_save_period == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        if (epoch + 1) % train_save_period == 0 and loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'chatbot.pth')
            print(f"Model saved at epoch {epoch+1} with the best loss: {best_loss:.4f}")
