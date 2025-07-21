import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pos_encoding(D, H, W):
    x = torch.arange(H)
    y = torch.arange(W)
    all_pos = []

    for j in range(0, D//4):
        x_sin = torch.sin(x / 10000**(4*j/D))
        all_pos.append(x_sin.repeat(W, 1).T.unsqueeze(0))
        x_cos = torch.cos(x / 10000**(4*j/D))
        all_pos.append(x_cos.repeat(W, 1).T.unsqueeze(0))

    for j in range(0, D//4):
        y_sin = torch.sin(y / 10000**(4*j/D))
        all_pos.append(y_sin.repeat(H, 1).unsqueeze(0))
        y_cos = torch.cos(y / 10000**(4*j/D))
        all_pos.append(y_cos.repeat(H, 1).unsqueeze(0))

    return torch.cat(all_pos)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, D, H_max, W_max):
        super(Encoder, self).__init__()
        self.pos_encoding = get_pos_encoding(D, H_max, W_max).to(DEVICE)

    def forward(self, x):
        size = x.shape
        H_, W_ = size[2], size[3]
        x = x + self.pos_encoding[:, 0:H_, 0:W_]
        return torch.flatten(x, start_dim=2).permute(0, 2, 1)


class Attention(nn.Module):
    def __init__(self, lstm_hidden_size, memory_size, att_hidden_size):
        super(Attention, self).__init__()

        self.W_1 = nn.Linear(2*lstm_hidden_size, att_hidden_size, bias=False)
        self.W_2 = nn.Linear(memory_size, att_hidden_size, bias=False)
        self.beta = nn.Linear(att_hidden_size, 1, bias=False)

    def alphas(self, decoder_hidden, encoder_outputs):
        decoder_hidden = self.W_1(decoder_hidden) # (batch, 1, hidden_size)
        encoder_outputs = self.W_2(encoder_outputs) # (batch, L, hidden_size)
        alpha = self.beta(F.tanh(decoder_hidden + encoder_outputs))
        return alpha # (batch, L, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        alpha = self.alphas(decoder_hidden, encoder_outputs) # (batch, L, 1)
        weights = F.softmax(alpha, dim=1).transpose(1, 2) #(batch, 1, L)
        context = torch.bmm(weights, encoder_outputs) #(batch, 1, memory_size)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, emb_size, lstm_hidden_size, att_hidden_size, o_hidden_size,  memory_size, output_size):
        super(Decoder, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.o_hidden_size = o_hidden_size

        self.lstm = nn.LSTM(emb_size + o_hidden_size, lstm_hidden_size, batch_first=True,
                           bidirectional=True)

        self.attention = Attention(lstm_hidden_size, memory_size, att_hidden_size)

        self.W_3 = nn.Linear(2*lstm_hidden_size + memory_size, o_hidden_size, bias=False)
        self.W_4 = nn.Linear(o_hidden_size, output_size, bias=False)

    def forward(self, x, hp, cp, op, encoder_outputs):
        lstm_input = torch.cat([x, op.unsqueeze(dim=1)], dim=2) #(batch, 1, input_size+hidden_size)
        hidden, (hn, cn) = self.lstm(lstm_input, (hp, cp))
        #hidden -> (batch, 1, 2*hidden_size)
        #(hn, cn) -> (2, batch, hidden_size)
        context, attention = self.attention(hidden, encoder_outputs)
        #context -> (batch, 1, memory_size)
        #attn_weights -> (batch, 1, L)
        o_input = torch.cat([hidden.squeeze(1), context.squeeze(1)], dim=1) #(batch, 2*hidden_size + memory_size)
        on = F.tanh(self.W_3(o_input))
        #next_o = (batch, hidden_size)
        output = self.W_4(on)
        return output, (hn, cn), on, attention


class Embedding(nn.Module):
    def __init__(self, input_size, embedded_size, padding_idx):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embedded_size, padding_idx=padding_idx)

    def forward(self, x):
        # x: symbols id -> (batch, time)
        embedded = self.embedding(x)
        # embedded -> (batch, time, embedded_size)
        return embedded


class Model(nn.Module):
    def __init__(self, cnn, encoder, embedding, decoder):
        super(Model, self).__init__()
        self.cnn = cnn
        self.encoder = encoder
        self.embedding = embedding
        self.decoder = decoder
        self.h_0 = nn.Linear(self.decoder.memory_size, 2*self.decoder.lstm_hidden_size)
        self.c_0 = nn.Linear(self.decoder.memory_size, 2*self.decoder.lstm_hidden_size)

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        cnn_outputs = self.cnn(x)

        encoder_outputs = self.encoder(cnn_outputs)

        B, L, D = encoder_outputs.shape

        hn = F.tanh(self.h_0(torch.mean(encoder_outputs, dim=1))) #(batch, 2*hidden)
        hn = hn.view(hn.shape[0], 2, hn.shape[1]//2).permute(1, 0, 2).contiguous() # (2, batch, decoder_hidden_size)


        cn = F.tanh(self.c_0(torch.mean(encoder_outputs, dim=1))) #(batch, 2*hidden)
        cn = hn.view(cn.shape[0], 2, cn.shape[1]//2).permute(1, 0, 2).contiguous() # (2, batch, decoder_hidden_size)
        on = torch.zeros(B, self.decoder.o_hidden_size, device=DEVICE) # (batch, decoder_hidden_size)

        next_token = y[:, 0] # (batch)

        all_outputs = torch.zeros(y.shape[0], y.shape[1], self.decoder.output_size, device=DEVICE) # (batch, len, vocab_size)
        for t in range(1, y.shape[1]):
            token_embedding = self.embedding(next_token).unsqueeze(1)
            output, (hn, cn), on, attn_weight = self.decoder(token_embedding, hn, cn, on, encoder_outputs)
            all_outputs[:, t, :] = output
            if teacher_forcing_ratio < torch.rand(1).item():
                next_token = torch.argmax(output, dim=1) # (batch)
            else:
                next_token = y[:, t] # (batch)
        return all_outputs


    def generate_output(self, x, start_token, max_len):
        cnn_outputs = self.cnn(x)

        encoder_outputs = self.encoder(cnn_outputs)

        B, L, D = encoder_outputs.shape

        hn = F.tanh(self.h_0(torch.mean(encoder_outputs, dim=1))) #(batch, 2*hidden)
        hn = hn.view(hn.shape[0], 2, hn.shape[1]//2).permute(1, 0, 2).contiguous() # (2, batch, decoder_hidden_size)


        cn = F.tanh(self.c_0(torch.mean(encoder_outputs, dim=1))) #(batch, 2*hidden)
        cn = hn.view(cn.shape[0], 2, cn.shape[1]//2).permute(1, 0, 2).contiguous() # (2, batch, decoder_hidden_size)
        on = torch.zeros(B, self.decoder.o_hidden_size, device=DEVICE) # (batch, decoder_hidden_size)

        next_token = torch.Tensor([start_token]).long().to(DEVICE)
        b = encoder_outputs.shape[0]
        next_token = next_token.repeat(b)
        preds = [next_token]

        for t in range(1, max_len):
            token_embedding = self.embedding(next_token).unsqueeze(1)
            output, (hn, cn), on, attn_weight = self.decoder(token_embedding, hn, cn, on, encoder_outputs)
            next_token = output.argmax(dim=1)
            preds.append(next_token)

        preds = torch.stack(preds, dim=1)
        return preds
