def optimize(model, optimizer, dataloader, nlp, vocab_size, epochs=1, iterations=None):
    from utils import get_tqdm, loopy
    tqdm = get_tqdm()

    model.train()
    if iterations is None: iterations = int(epochs * len(dataloader['train']))
    prog_bar = tqdm(range(iterations))
    gen = loopy(dataloader['train'])
    losses = []
    
    for batch in prog_bar:
        feature, caption = next(gen)
        loss = get_loss(model, feature, caption[0], nlp, vocab_size)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        prog_bar.set_description(f'{loss.item():.2f}')
        if not batch % 100: losses.append(loss.item())
        
    return losses

def get_loss(model, feature, caption, nlp, vocab_size):
    import magnet as mag
    from torch.nn import functional as F
    
    cap, target = process_caption(caption, nlp, vocab_size)    
    y = model(feature.to(mag.device), cap.to(mag.device))
    return F.cross_entropy(y.squeeze(0), target.to(mag.device))

def process_caption(caption, nlp, vocab_size):
    import torch
    from numpy import stack
    from nlp import word_idx

    caption = nlp(caption)
    vectors = torch.tensor(stack([token.vector for token in caption[:-1]])).unsqueeze(0)
    indices = torch.tensor([word_idx(token, nlp, vocab_size) for token in caption[1:]])
    return vectors, indices