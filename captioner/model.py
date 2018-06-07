import magnet as mag

import torch
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, feature_dim, embedding_dim, hidden_size, num_layers, rnn_type, vocab_size):
        super().__init__()
        self.fc_feat = nn.Linear(feature_dim, hidden_size * num_layers)
        rnn_module_dict = {'lstm': nn.LSTM, 'rnn': nn.RNN}
        self.rnn = rnn_module_dict[rnn_type](embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size + 1)
        self.to(mag.device)

    def forward(self, feat, cap=None, nlp=None, max_len=20):
        h0 = F.relu(self.fc_feat(feat)).view(1, self.rnn.num_layers, -1).transpose(0, 1).contiguous()
        if isinstance(self.rnn, nn.LSTM): h0 = (h0, torch.zeros_like(h0))

        if cap is None:
            if self.training: raise ValueError('Provide caption while training')
            return self._generate(h0, nlp, max_len)

        x = self.rnn(cap, h0)[0]
        return self.fc(x)

    def _generate(self, h, nlp, max_len):
        y = torch.zeros(1, max_len).to(self.device)
        x = torch.tensor(nlp('`')[0].vector).view(1, 1, -1).to(self.device)

        for i in range(max_len): y[0, i], h, x = self._sample(x, h, nlp)

        return y

    def _sample(self, x, h, nlp):
        x, h = self.rnn(x, h)
        x = self.fc(x).squeeze(1)
        idx = x.max(-1)[1]

        if idx == self.fc.out_features - 1:
            next_vector = torch.zeros(1, 1, embedding_dim).to(self.device)
        else:
            hash_val = nlp.vocab.vectors.find(row=idx.item())[0]
            next_vector = torch.tensor(nlp.vocab.vectors[hash_val]).to(self.device).view(1, 1, -1)

        return idx, h, next_vector

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        try: self.device = next(self.parameters())[0].device
        except: pass
        return self