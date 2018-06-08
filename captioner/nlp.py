import torch

from numpy import stack

def get_nlp(model, vocab_size, save_path=None):
	import spacy

	nlp = spacy.load(model)
	nlp.pipeline = []
	set_vocab_size(nlp, vocab_size, save_path)
	return nlp

def set_vocab_size(nlp, vocab_size, save_path=None):
    if save_path is not None and save_path.exists():
        nlp.vocab.from_disk(save_path)
        new_vocab_size = nlp.vocab.vectors.shape[0]
        if new_vocab_size != vocab_size: prune_vocab(nlp, save_path)
    else: prune_vocab(nlp, save_path)

def prune_vocab(nlp, save_path=None):
    nlp.vocab.prune_vectors(vocab_size, prune_batch_size)
    if save_path is not None: nlp.vocab.to_disk(save_path)

def word_idx(word, nlp, oov):
    if isinstance(word, str): word = nlp(word)[0]
    idx = int(nlp.vocab.vectors.find(key=word.orth))
    if idx == -1: return oov
    return idx

def idx_word(idx, nlp):
    hash_code = nlp.vocab.vectors.find(row=idx)
    if len(hash_code) == 0: return '<UNK>'

    return nlp.vocab.strings[hash_code[0]]

def process_caption(caption, nlp, vocab_size):
    caption = nlp(caption)
    vectors = torch.tensor(stack([token.vector for token in caption[:-1]])).unsqueeze(0)
    indices = torch.tensor([word_idx(token, nlp, vocab_size) for token in caption[1:]])
    return vectors, indices