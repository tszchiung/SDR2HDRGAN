from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
import os
from keras.models import load_model
from scipy.misc import imresize
import argparse

import Utils, Utils_model
from Utils_model import VGG_LOSS

image_shape = (384, 384, 3) # shape

def test_model(input_hdr, model, number_of_images, output_dir):
    x_test_sdr, x_test_hdr = Utils.load_test_data_for_model(input_hdr, 'jpg', number_of_images)
    Utils.plot_test_generated_images_for_model(output_dir, model, x_test_hdr, x_test_sdr)

def test_model_for_sdr_images(input_sdr, model, number_of_images, output_dir):
    x_test_sdr = Utils.load_test_data(input_sdr, 'jpg', number_of_images)
    Utils.plot_test_generated_images(output_dir, model, x_test_sdr)

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ihdr', '--input_hdr', action='store', dest='input_hdr', default='./data_hdr/' ,
                    help='Path for input HDR images')                    
    parser.add_argument('-isdr', '--input_sdr', action='store', dest='input_sdr', default='./data_sdr/' ,
                    help='Path for input SDR images')                    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/' ,
                    help='Path for Output images')    
    parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default='./model/gen_model3000.h5' ,
                    help='Path for model')                    
    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=25 ,
                    help='Number of Images', type=int)                    
    parser.add_argument('-t', '--test_type', action='store', dest='test_type', default='test_model',
                    help='Option to test model output or to test low resolution image')
    values = parser.parse_args()
    
    loss = VGG_LOSS(image_shape)  
    model = load_model(values.model_dir , custom_objects={'vgg_loss': loss.vgg_loss})
    
    if values.test_type == 'test_model':
        test_model(values.input_hdr, model, values.number_of_images, values.output_dir)
        
    elif values.test_type == 'test_sdr_images':
        test_model_for_sdr_images(values.input_sdr, model, values.number_of_images, values.output_dir)
        
    else:
        print("No such option")
