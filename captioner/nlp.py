import torch

from numpy import stack

def get_nlp(model, vocab_size, save_path=None, prune_batch_size=8):
	"""
	This method loads the model and then [_prunes_](https://spacy.io/api/vocab#prune_vectors) it to the vocab_size.
	It retains the most frequent tokens and maps out of vocabulary words at runtime to synonyms.

	Since this pruning takes time, the resultant vocabulary is saved as a checkpoint.

	On reruns, this method loads the checkpoint if found, and then checks if the vocabulary size is the same as that requested.
	If not, it prunes the model as before and overwrites the existing checkpoint.

	:param model: The spaCy model to use. The larger the better.
	:param vocab_size: The size of the vocabulary to be used. The larger the size, the more rare words will be covered. This should not be as much of an issue since
						SpaCy automatically replaces unknown words with their synonyms at runtime. Beware though, that a larger size will consume tons of memory since
						we'll be doing a softmax on the scores.
	:param save_path: The path where the pruned vocabulary will be checkpointed.
	:return: The requested spaCy model with the desired vocabulary size.
	"""
	import spacy

	nlp = spacy.load(model)
	nlp.pipeline = []
	_set_vocab_size(nlp, vocab_size, prune_batch_size, save_path)
	return nlp

def _set_vocab_size(nlp, vocab_size, prune_batch_size, save_path=None):
	if save_path is not None and save_path.exists():
		nlp.vocab.from_disk(save_path)
		new_vocab_size = nlp.vocab.vectors.shape[0]
		if new_vocab_size != vocab_size: _prune_vocab(nlp, vocab_size, prune_batch_size, save_path)
	else: _prune_vocab(nlp, vocab_size, prune_batch_size, save_path)

def _prune_vocab(nlp, vocab_size, prune_batch_size, save_path=None):
	nlp.vocab.prune_vectors(vocab_size, prune_batch_size)
	if save_path is not None: nlp.vocab.to_disk(save_path)

def word_idx(word, nlp):
	"""
	Converts the word to it's corresponding index in the vocubalary of the model.

	:param word: The word token (string or spaCy token).
	:param nlp: The spaCy model.
	:return: Index of the word in the vocabulary.
	"""
	if isinstance(word, str): word = nlp(word)[0]
	idx = int(nlp.vocab.vectors.find(key=word.orth))
	if idx == -1: return nlp.vocab.vectors.shape[0]
	return idx

def idx_word(idx, nlp):
	"""
	Returns the token/word at the specified index of the spaCy model.

	:param idx: The index to query.
	:param nlp: The spaCy model.
	:return: Word at the index in the vocabulary.
	"""
	hash_code = nlp.vocab.vectors.find(row=idx)
	if len(hash_code) == 0: return '<UNK>'

	return nlp.vocab.strings[hash_code[0]]

def process_caption(caption, nlp):
	"""
	Returns the word vectors and corresponding indices of the caption.

	:param caption: The caption to process.
	:param nlp: The spaCy model.
	:return: A vectors, indices tuple
	"""
	caption = nlp(caption)
	vectors = torch.tensor(stack([token.vector for token in caption[:-1]])).unsqueeze(0)
	indices = torch.tensor([word_idx(token, nlp) for token in caption[1:]])
	return vectors, indices