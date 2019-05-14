from Network import Generator, Discriminator
import Utils_model, Utils
from Utils_model import VGG_LOSS

from keras.models import Model
from keras.layers import Input
from tqdm import tqdm
import numpy as np
import argparse

np.random.seed(10)
image_shape = (384, 384, 3) # shape depends on training samples

def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan

def train(epochs, batch_size, input_dir, output_dir, model_save_dir, number_of_images, train_test_ratio):
    
    x_train_sdr, x_train_hdr, x_test_sdr, x_test_hdr = Utils.load_training_data(input_dir, '.jpg', number_of_images, train_test_ratio) 
    loss = VGG_LOSS(image_shape)  
    
    batch_count = int(x_train_hdr.shape[0] / batch_size)
    
    generator = Generator(image_shape).generator()
    discriminator = Discriminator(image_shape).discriminator()

    optimizer = Utils_model.get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    
    gan = get_gan_network(discriminator, image_shape, generator, optimizer, loss.vgg_loss)
    
    loss_file = open(model_save_dir + 'losses.txt' , 'w+')
    loss_file.close()

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            
            rand_nums = np.random.randint(0, x_train_hdr.shape[0], size=batch_size)
            
            image_batch_hdr = x_train_hdr[rand_nums]
            image_batch_sdr = x_train_sdr[rand_nums]
            generated_images_hdr = generator.predict(image_batch_sdr)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2
            
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(image_batch_hdr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_hdr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            
            rand_nums = np.random.randint(0, x_train_hdr.shape[0], size=batch_size)
            image_batch_hdr = x_train_hdr[rand_nums]
            image_batch_sdr = x_train_sdr[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(image_batch_sdr, [image_batch_hdr,gan_Y])
            
            
        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)
        
        loss_file = open(model_save_dir + 'losses.txt' , 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, gan_loss, discriminator_loss) )
        loss_file.close()

        if e == 1 or e % 5 == 0:
            Utils.plot_generated_images(output_dir, e, generator, x_test_hdr, x_test_sdr)
        if e % 500 == 0:
            generator.save(model_save_dir + 'gen_model%d.h5' % e)
            discriminator.save(model_save_dir + 'dis_model%d.h5' % e)


if __name__== "__main__":
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir', default='./data/' ,
                    help='Path for input images')                    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/' ,
                    help='Path for Output images')    
    parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir', default='./model/' ,
                    help='Path for model')
    parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=64,
                    help='Batch Size', type=int)                    
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=1000 ,
                    help='number of iteratios for trainig', type=int)                    
    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=1000 ,
                    help='Number of Images', type= int)                    
    parser.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.8 ,
                    help='Ratio of train and test Images', type=float)    
    values = parser.parse_args()
    
    train(values.epochs, values.batch_size, values.input_dir, values.output_dir, values.model_save_dir, values.number_of_images, values.train_test_ratio)
