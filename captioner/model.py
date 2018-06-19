import magnet as mag

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from captioner.nlp import idx_word

class Model(nn.Module):
    def __init__(self, feature_dim, embedding_dim, hidden_size, num_layers, rnn_type, vocab_size):
        """
        The generative RNN model which produces the captions.

        The image features are passed through a Linear layer, reshaped and fed to the RNN as it's initial hidden state.

        The RNN's hidden states for each timestep are passed through another Linear layer to get the scores for each
        word for each timestep.

        While sampling, teacher forcing is disabled and the RNN proceeds in a step-by-step fashion.
        The previous output is obtained by beam search on the scores, and used as current input.

        Overall, beam search returns the most likely captions.

        :param feature_dim: The dimensionality of the CNN features.
        :param embedding_dim: The dimensionality of the word embeddings.
        :param hidden_size: The hidden size for each layer of the RNN.
        :param num_layers: Number of layers in the RNN.
        :param rnn_type: The type of RNN to use.
                        Options: ['LSTM', 'RNN']
        :param vocab_size: The vocabulary size used.
        """
        from captioner.utils import BeamSearch

        super().__init__()
        self.fc_feat = nn.Linear(feature_dim, hidden_size * num_layers)
        rnn_module_dict = {'LSTM': nn.LSTM, 'RNN': nn.RNN}
        self.rnn = rnn_module_dict[rnn_type](embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size + 1)

        self._search = BeamSearch()

        self.to(mag.device)

    def forward(self, features, cap=None, nlp=None, beam_size=3, max_len=20, probabilistic=False):
        """
        Returns the scores/captions of image features.

        :param features: The CNN extracted image features.
        :param cap: The caption string. Cannot be left out during training, since we're teacher forcing.
                    During evaluation, if left out, uses beam search to generate captions.
        :param nlp: spaCy model used for tokenization and vectorization.
        :param beam_size: Captions are sampled using beam search with this size.
        :param max_len: The maximum allowed length upto which captions are generated. It's then pruned upto the first sentence.
        :param probabilistic: If True, the beam search retains nodes at each iteration according to their probabilities.
        :return: If training, returns the scores for each word at each timestep.
                If evaluating, returns a likely caption.
        """
        if cap is None:
            if self.training: raise ValueError('Provide caption while training')
            return self._generate(features, nlp, beam_size, max_len, probabilistic)

        h0 = self._get_initial_hidden(features)
        x = self.rnn(cap, h0)[0]
        return self.fc(x)

    def _generate(self, features, nlp, beam_size, max_len, probabilistic):
        self._search.build = lambda *args: self._build(*args, nlp=nlp)
        branches = self._search(beam_size, features, max_len, probabilistic)

        captions, probs = [], []
        for branch in branches:
            indices, prob = branch.content, branch.score
            caption = ' '.join([idx_word(idx, nlp) for idx in indices])
            caption = caption[:caption.find('.')]
            if caption not in captions:
                captions.append(caption)
                probs.append(prob)

        probs = np.array(probs)
        sort_idx = np.argsort(probs)[::-1]
        probs = list(probs / probs.sum())

        return [(captions[i], probs[i]) for i in sort_idx]

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
        h0 = F.relu(self.fc_feat(features)).view(1, -1, self.rnn.hidden_size).transpose(0, 1).contiguous()
        if isinstance(self.rnn, nn.LSTM): h0 = (h0, torch.zeros_like(h0))

        return h0

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        try: self.device = next(self.parameters())[0].device
        except: pass
        return self