import re
import torch

def Tokenizer(user_input, model_output, pad_token="<PAD>", unk_token="<UNK>", start_token="<SOS>", end_token="<EOS>"):
    user_input_sentences = [re.sub(r'[^\w\s]', r' \g<0> ', sentence.lower()) for sentence in user_input]
    model_output_sentences = [re.sub(r'[^\w\s]', r' \g<0> ', sentence.lower()) for sentence in model_output]
    words = ' '.join(user_input_sentences + model_output_sentences).split()
    special_tokens = [pad_token, unk_token, start_token, end_token]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
    vocab_size = len(vocab)

    enc_input_sentences = [sentence.split() for sentence in user_input_sentences]
    dec_input_sentences = [[start_token] + sentence.split() for sentence in model_output_sentences]
    dec_target_sentences = [sentence.split() + [end_token] for sentence in model_output_sentences]
    enc_max_len = max(len(sent) for sent in enc_input_sentences)
    dec_max_len = max(len(sent) for sent in dec_input_sentences)

    tokenized_enc_input = [
        [vocab[word] for word in sent] + ([vocab[pad_token]] * (enc_max_len - len(sent)))
        for sent in enc_input_sentences
    ]
    tokenized_dec_input = [
        [vocab[word] for word in sent] + ([vocab[pad_token]] * (dec_max_len - len(sent)))
        for sent in dec_input_sentences
    ]
    tokenized_dec_target = [
        [vocab[word] for word in sent] + ([vocab[pad_token]] * (dec_max_len - len(sent)))
        for sent in dec_target_sentences
    ]
    enc_input_tensor = torch.tensor(tokenized_enc_input)
    dec_input_tensor = torch.tensor(tokenized_dec_input)
    dec_target_tensor = torch.tensor(tokenized_dec_target)
    return enc_input_tensor, dec_input_tensor, dec_target_tensor, vocab_size, enc_max_len, dec_max_len, vocab
