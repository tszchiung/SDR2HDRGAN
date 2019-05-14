from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
from scipy.misc import imresize
import os
import sys

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],input_shape[1] * scale,input_shape[2] * scale,int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape
    
    def subpixel(x):
        return tf.depth_to_space(x, scale)
        
    return Lambda(subpixel, output_shape=subpixel_shape)
    
# Takes list of images and provide HDR images in form of numpy array
def hdr_images(images):
    images_hdr = array(images)
    return images_hdr

# Takes list of images and provide SDR images in form of numpy array
def sdr_images(images):
    images_sdr = array(images)
    return images_sdr
    
def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories
    
def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = data.imread(os.path.join(d,f))
                if len(image.shape) > 2:
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                count = count + 1
    return files     

def load_data(directory, ext):
    files = load_data_from_dirs(load_path(directory), ext)
    return files
    
def load_training_data(directory, ext, number_of_images = 1000, train_test_ratio = 0.8):
    number_of_train_images = int(number_of_images * train_test_ratio)   
    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    test_array = array(files)
    if len(test_array.shape) < 3:
        print("Images are of not same shape")
        print("Please provide same shape images")
        sys.exit()
    
    x_train = files[:number_of_train_images]
    x_test = files[number_of_train_images:number_of_images]
    
    x_train_hdr = hdr_images(x_train)
    x_train_hdr = normalize(x_train_hdr)
    
    x_train_sdr = sdr_images(x_train)
    x_train_sdr = normalize(x_train_sdr)
    
    x_test_hdr = hdr_images(x_test)
    x_test_hdr = normalize(x_test_hdr)
    
    x_test_sdr = sdr_images(x_test)
    x_test_sdr = normalize(x_test_sdr)
    
    return x_train_sdr, x_train_hdr, x_test_sdr, x_test_hdr


def load_test_data_for_model(directory, ext, number_of_images = 100):
    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    x_test_hdr = hdr_images(files)
    x_test_hdr = normalize(x_test_hdr)
    
    x_test_sdr = sdr_images(files)
    x_test_sdr = normalize(x_test_sdr)
    
    return x_test_sdr, x_test_hdr
    
def load_test_data(directory, ext, number_of_images = 100):
    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    x_test_sdr = sdr_images(files)
    x_test_sdr = normalize(x_test_sdr)
    
    return x_test_sdr
    
# While training save generated image(in form SDR, gen_HDR, HDR)
# Save only one image as sample  
def plot_generated_images(output_dir, epoch, generator, x_test_hdr, x_test_sdr , dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hdr.shape[0]
    print(examples)
    value = randint(0, examples)
    image_batch_hdr = denormalize(x_test_hdr)
    image_batch_sdr = x_test_sdr
    gen_img = generator.predict(image_batch_sdr)
    generated_image = denormalize(gen_img)
    image_batch_sdr = denormalize(image_batch_sdr)
    
    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_sdr[value], interpolation='nearest')
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], interpolation='nearest')
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hdr[value], interpolation='nearest')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)
    
    #plt.show()
    
# Plots and save generated images(in form SDR, gen_HDR, HDR) from model to test the model 
# Save output for all images given for testing  
def plot_test_generated_images_for_model(output_dir, generator, x_test_hdr, x_test_sdr , dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hdr.shape[0]
    image_batch_hdr = denormalize(x_test_hdr)
    image_batch_sdr = x_test_sdr
    gen_img = generator.predict(image_batch_sdr)
    generated_image = denormalize(gen_img)
    image_batch_sdr = denormalize(image_batch_sdr)
    
    for index in range(examples):
    
        plt.figure(figsize=figsize)
    
        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(image_batch_sdr[index], interpolation='nearest')
        plt.axis('off')
        
        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')
    
        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(image_batch_hdr[index], interpolation='nearest')
        plt.axis('off')
    
        plt.tight_layout()
        plt.savefig(output_dir + 'test_generated_image_%d.png' % index)
    
        #plt.show()

# Takes SDR images and save respective HDR images
def plot_test_generated_images(output_dir, generator, x_test_sdr, figsize=(5, 5)):
    
    examples = x_test_sdr.shape[0]
    image_batch_sdr = denormalize(x_test_sdr)
    gen_img = generator.predict(image_batch_sdr)
    generated_image = denormalize(gen_img)
    
    for index in range(examples):
    
        #plt.figure(figsize=figsize)
    
        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir + 'high_res_result_image_%d.png' % index)
    
        #plt.show()
