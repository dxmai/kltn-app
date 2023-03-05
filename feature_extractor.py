import cv2
import numpy as np
import tensorflow as tf
from img2vec_pytorch import Img2Vec

def CalHistogram(img, bin=[8,8,8]):
    img = img[:, :, ::-1]
    img = np.array(img)
    hist = cv2.calcHist([img],[0, 1, 2],None,[bin[0], bin[1], bin[2]],[0,256, 0, 256, 0, 256])
    hist = hist.reshape(1, -1)/ hist.sum()
    return hist

def split_image_to_tiles(img, rows=3, columns=3):
    height = img.shape[0]
    width = img.shape[1]
    h = height // rows
    w = width // columns
    tiles = []
    hindex = 0
    for i in range(0, height, h):  
        windex = 0
        for j in range(0, width, w):
            addw = j + w if windex != columns - 1 else width
            addh = i + h if hindex != rows - 1 else height
            tile = img[i:addh, j:addw, :]
            tiles.append(tile)
            if windex == columns - 1: 
                break
            windex += 1
        if hindex == rows - 1:
            break
        hindex += 1
    return tiles

def get_region_histogram(img, rows=3, columns=3):
    tiles = split_image_to_tiles(img, rows, columns)
    features = []
    for tile in tiles:
        histogram = CalHistogram(tile, [8,8,8])
        histogram = cv2.normalize(histogram, histogram).flatten()
        features.extend(histogram)
    features = np.array(features)
    features = features.reshape(1, -1)
    return features

def get_mobilenet(img):
    IMAGE_DIMS = (160, 160, 3)
    mnet_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMAGE_DIMS,
                                               include_top=False, weights='imagenet')
    img=cv2.resize(img, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    im_toarray = tf.keras.preprocessing.image.img_to_array(img)
    im_toarray = np.expand_dims(img, axis=0)
    im_toarray = tf.keras.applications.mobilenet.preprocess_input(im_toarray)
    data_stack = np.vstack([im_toarray]) 
    feature = mnet_model.predict(data_stack)
    return feature

def CalHistogram_RGB(img):
    features = []
    for channel in range(0, 3):
        hist = cv2.calcHist([img], [channel], None, [255], [1, 256])
        features.append(hist)
    features = np.array(features)
    res = features.reshape(1, features.shape[0] * features.shape[1] * features.shape[2])
    return res

def get_dominant_color(img, number_of_colors=5):
    nrow, ncol, nchl = img.shape
    g = img.reshape(nrow*ncol,nchl)

    flags = cv2.KMEANS_RANDOM_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    # Add one more color in case dominant color is black
    _, labels, centroids = cv2.kmeans(np.float32(g), number_of_colors + 1, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette

def check_color(palette, number_of_colors=5):
    result = []
    for color in palette:
        if color[0] <= 32 and color[1] <= 32 and color[2] <= 32:
            continue
        result.append(color)
        if len(result) == number_of_colors:
            break
    result = np.array(result)
    while result.shape[0] < 5:
        result = np.append(result, [[0, 0, 0]], axis=0)
    result = result.reshape(1, -1)
    return result

def get_resnet_embedding(img):
    img2vec = Img2Vec(cuda=True)
    vec = img2vec.get_vec(img, tensor=True)
    vec = vec.reshape([-1, 1])
    vec = vec.detach().numpy()
    return vec
