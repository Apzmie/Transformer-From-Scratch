from tokenizer import tokenizer
from transformer import Transformer
import re
import pandas as pd
import torch

def generate(data, eval_input, d_model, num_heads, num_layers, max=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.DataFrame(data)
    user_input = df['user_input'].tolist()
    model_output = df['model_output'].tolist()
    _, _, _, vocab_size, enc_max_len, dec_max_len, vocab = tokenizer(user_input, model_output)
    model = Transformer(d_model, num_heads, num_layers, vocab_size, enc_max_len, dec_max_len)
    model.load_state_dict(torch.load('chatbot.pth'))
    model.to(device)
    model.eval()

    idx_to_word = {i: w for w, i in vocab.items()}
    sentence = eval_input.lower()
    sentence = re.sub(r'[^\w\s]', r' \g<0> ', sentence)
    input_tokens = sentence.split()
    input_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in input_tokens]
    if len(input_ids) < enc_max_len:
        input_ids += [vocab["<PAD>"]] * (enc_max_len - len(input_ids))
    else:
        input_ids = input_ids[:enc_max_len]     
    enc_input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    dec_input = [vocab["<SOS>"]]

    if max is None:
        max = dec_max_len

    for _ in range(max):
        dec_input_tensor = torch.tensor([dec_input], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(enc_input_tensor, dec_input_tensor, vocab)
        next_token_logits = output[0, -1]
        next_token_id = torch.argmax(next_token_logits).item()
        if next_token_id == vocab["<EOS>"]:
            break
        dec_input.append(next_token_id)

    result_tokens = [idx_to_word[token_id] for token_id in dec_input[1:]]
    result = ' '.join(result_tokens)
    result = re.sub(r'\s+([?.!,])', r'\1', result)
    result = result.capitalize()
    return result

def chat(data, d_model, num_heads, num_layers):
    print("Enter 'exit' to quit.")
    while True:
        eval_input = input("You: ") 
        if eval_input.lower() in ["exit"]:
            print("Exit")
            break        
        response = generate(data, eval_input, d_model, num_heads, num_layers)
        print("Bot:", response)
