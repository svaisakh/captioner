from contextlib import contextmanager

def show_coco(img, captions):
    import matplotlib.pyplot as plt

    from numpy.random import randint
    
    captions = captions[randint(len(captions))]
    
    def show_image(image, title):
        image = image.permute(1, 2, 0).numpy().copy()
        i_min, i_max = image.min(), image.max()
        image = (image - i_min) / (i_max - i_min) 
        plt.imshow(image)
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.title(title)
        plt.show()

    for i, c in zip(img, captions): show_image(i, c)
        

def working_directory(path):
    """path can also be a function in case of decorator"""
    
    from inspect import isfunction
    if not isfunction(path):
        return _working_directory_context_manager(path)
    
    from functools import wraps

    @wraps(path)
    def new_fn(*args, **kwargs):
        from pathlib import PosixPath
        
        working_path = [a for a in args if type(a) is PosixPath]
        if len(working_path) != 0: working_path = working_path[0]
        else: 
            working_path = [v for v in kwargs.values() if type(v) is PosixPath]
            if len(working_path) != 0: working_path = working_path[0]
            else: raise RuntimeError('No suitable paths found')

        with _working_directory_context_manager(working_path): 
            return path(*args, **kwargs)
            
    return new_fn

@contextmanager
def _working_directory_context_manager(path):
    import os
    
    # Change to working directory
    path_cwd = os.getcwd()
    os.chdir(path)
    
    yield
    
    os.chdir(path_cwd) # Change back to working directory

def get_tqdm():
    import tqdm

    try:
        get_ipython
        return getattr(tqdm, 'tqdm_notebook')
    except:
        return getattr(tqdm, 'tqdm')