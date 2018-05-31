# Extractor
architecture = 'resnet34'
extractor_batch_size = 8
image_shape = 256

if type(image_shape) is int: image_shape = (image_shape, image_shape)