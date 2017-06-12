from scipy import io, ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

####### DEFINE USEFUL CONSTANTS HERE #######

model_path = 'data/imagenet-vgg-verydeep-19.mat'

OUTPUT_DIR = 'out/'

CONTENT_IMG = 'data/special-k.jpg'
STYLE_IMG = 'data/watercolor1.jpg'

IMG_WIDTH = 960 
IMG_HEIGHT = 960 
NUM_CHANNELS = 3

MEAN_PIXELS = np.array([123.68, 116.779, 103.939], dtype='float32').reshape((1,1,1,3))

# Content/Style tradeoff parameters
alpha = 100
beta = 5

# Ratio of img to noise
NOISE_LVL = 0.6

# Run the model for this many iterations
NUM_ITERS = 1000

############################################

def load_img(path, mean=MEAN_PIXELS,  preprocess=True):
    img = ndimage.imread(path).astype('float32')
    img = misc.imresize(img, (IMG_HEIGHT, IMG_WIDTH)).astype('float32')
    if preprocess:
        img = np.reshape(img, ((1,) + img.shape))
        img -= mean
    return img

def save_img(img, path, mean=MEAN_PIXELS, postprocess=True):
    if postprocess:
        img += mean
    img = img[0]
    misc.imsave(path, img)

def generate_noise_image(content_img, nc=NUM_CHANNELS, nl=NOISE_LVL):
    img = np.random.uniform(
          -20, 20,
          (1, content_img.shape[1], content_img.shape[0], nc)).astype('float32')
    img = img * nl + content_img * (1 - nl)
    return img 

def plot_img(img):
    imshow(img)

def load_vggNet(path):
    model_dict = io.loadmat(path)
    model = _load_model_from_dict(model_dict)
    return model

def get_model_keys(model_dict):
    return model_dict.keys()

def _load_model_from_dict(model_dict):
    layers = model_dict['layers'][0]
    layer_names = [layers[i][0][0][0][0] for i in range(len(layers))]
    vgg_layer_list = [None] * len(layer_names)
    for i, name in enumerate(layer_names):
        vgg_layer_list[i] = {} 
        vgg_layer_list[i]['name'] = name 
        vgg_layer_list[i]['type'] = layers[i][0][0][1][0]
        if vgg_layer_list[i]['type'] not in ['relu', 'softmax']:
            vgg_layer_list[i]['W'] = layers[i][0][0][2][0][0]
            vgg_layer_list[i]['b'] = layers[i][0][0][2][0][1]
        else:
            # vgg_layer_list[i]['W'] = np.array([0])
            # vgg_layer_list[i]['b'] = np.array([0])
            continue
    return vgg_layer_list
