import magnet as mag
import torch

def optimize(model, optimizer, history, dataloader, nlp, vocab_size, save_path, epochs=1, iterations=None, save_every=5, write_every=1):
    from captioner.utils import get_tqdm, loopy
    from time import time

    tqdm = get_tqdm()
    start_time = time()
    mean = lambda x: sum(x) / len(x)

    model.train()
    if iterations is None: iterations = int(epochs * len(dataloader['train']))
    prog_bar = tqdm(range(iterations))
    gen = {mode: loopy(dataloader[mode]) for mode in ('train', 'val')}
    running_history = {'loss': []}

    for batch in prog_bar:
        feature, caption = next(gen['train'])
        loss = get_loss(model, feature, caption[0], nlp, vocab_size)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_history['loss'].append(loss.item())
        history['iterations'] += 1

        if (time() - start_time > write_every * 60) or (batch == iterations - 1):
            mean_loss = mean(running_history['loss'])
            history['loss'].append(mean_loss)
            running_history['loss'] = []

            feature, caption = next(gen['val'])
            with mag.eval(model): loss = get_loss(model, feature, caption[0], nlp, vocab_size).item()
            history['val_loss'].append(loss)

            prog_bar.set_description(f'{mean_loss:.2f} val={loss:.2f}')
            
        if (time() - start_time > save_every * 60) or (batch == iterations - 1):
            torch.save(model.state_dict(), save_path / 'model.pt')
            torch.save(optimizer.state_dict(), save_path  / 'optimizer.pt')

def get_loss(model, feature, caption, nlp, vocab_size):
    from torch.nn import functional as F

    cap, target = process_caption(caption, nlp, vocab_size)
    y = model(feature.to(mag.device), cap.to(mag.device))
    return F.cross_entropy(y.squeeze(0), target.to(mag.device))

def process_caption(caption, nlp, vocab_size):
    from numpy import stack
    from captioner.nlp import word_idx

    caption = nlp(caption)
    vectors = torch.tensor(stack([token.vector for token in caption[:-1]])).unsqueeze(0)
    indices = torch.tensor([word_idx(token, nlp, vocab_size) for token in caption[1:]])
    return vectors, indices