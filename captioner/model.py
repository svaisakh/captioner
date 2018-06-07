import magnet as mag

import torch
from torch import nn
from torch.nn import functional as F

from captioner.nlp import idx_word

class Model(nn.Module):
    def __init__(self, feature_dim, embedding_dim, hidden_size, num_layers, rnn_type, vocab_size):
        from captioner.utils import BeamSearch

        super().__init__()
        self.fc_feat = nn.Linear(feature_dim, hidden_size * num_layers)
        rnn_module_dict = {'lstm': nn.LSTM, 'rnn': nn.RNN}
        self.rnn = rnn_module_dict[rnn_type](embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size + 1)

        self._search = BeamSearch()

        self.to(mag.device)

    def forward(self, features, cap=None, nlp=None, beam_size=3, max_len=20):
        if cap is None:
            if self.training: raise ValueError('Provide caption while training')
            return self._generate(features, nlp, beam_size, max_len)

        h0 = self._get_initial_hidden(features)
        x = self.rnn(cap, h0)[0]
        return self.fc(x)

    def _generate(self, features, nlp, beam_size, max_len):
        self._search.build = lambda *args: self._build(*args, nlp=nlp)
        branches = self._search(beam_size, features, max_len)

        captions, probs = [], []
        for branch in branches:
            indices, prob = branch.content, branch.score
            caption = ' '.join([idx_word(idx, nlp) for idx in indices])
            caption = caption[:caption.find('.')]
            if caption not in captions:
                captions.append(caption)
                probs.append(prob)

        return list(zip(captions, probs))

    def _build(self, content, context, nlp):
        prev_ids = content
        if len(prev_ids) == 0:
            x = torch.tensor(nlp('`')[0].vector).view(1, 1, -1).to(self.device)
            h = self._get_initial_hidden(context)
        else:
            h = context
            last_idx = prev_ids[-1]
            if last_idx == self.fc.out_features - 1:
                x = torch.zeros(1, 1, self.rnn.input_size).to(self.device)
            else:
                hash_val = nlp.vocab.vectors.find(row=last_idx)[0]
                x = torch.tensor(nlp.vocab.vectors[hash_val]).to(self.device).view(1, 1, -1)

        x, context = self.rnn(x, h)
        log_probs = F.log_softmax(self.fc(x).view(-1), dim=-1)
        content, scores  = list(zip(*[(i, p.item()) for i, p in enumerate(log_probs)]))

        return content, scores, context

    def _get_initial_hidden(self, features):
        h0 = F.relu(self.fc_feat(features)).view(1, self.rnn.num_layers, -1).transpose(0, 1).contiguous()
        if isinstance(self.rnn, nn.LSTM): h0 = (h0, torch.zeros_like(h0))

        return h0

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        try: self.device = next(self.parameters())[0].device
        except: pass
        return self