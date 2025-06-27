import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import os
from PIL import Image
from tqdm import tqdm
import random
import os.path
import imageio
from utils import *
from models.models import *
import tensorflow as tf


input_size = (256,256,1)

SCANNED_PATH = "data/train/scanned/"
GROUND_TRUTH_PATH = "data/train/ground_truth/"


def train_gan(generator,discriminator, ep_start=1, epochs=1, batch_size=128):
    
    list_deg_images= [f for f in os.listdir(SCANNED_PATH) if f.endswith(".png")]
    list_clean_images= [f for f in os.listdir(SCANNED_PATH) if f.endswith(".png")]
    
    list_deg_images.sort()
    list_clean_images.sort()

    gan = get_gan_network(discriminator, generator)

    # TensorBoard setup
    log_dir = "logs/gan_training"
    discriminator_log_dir = os.path.join(log_dir, "discriminator")
    generator_log_dir = os.path.join(log_dir, "generator")

    os.makedirs(discriminator_log_dir, exist_ok=True)
    os.makedirs(generator_log_dir, exist_ok=True)

    # Summary writers for TensorBoard
    discriminator_writer = tf.summary.create_file_writer(discriminator_log_dir)
    generator_writer = tf.summary.create_file_writer(generator_log_dir)

    global_step = 0
    
    for e in range(ep_start, epochs+1):
        print ('\n Epoch:' ,e)
        
        for im in tqdm(range (len(list_deg_images))):
            


            deg_image_path = (SCANNED_PATH+list_deg_images[im])
            deg_image = Image.open(deg_image_path)# /255.0
            deg_image = deg_image.convert('L')
            deg_image.save('curr_deg_image.png')

            deg_image = plt.imread('curr_deg_image.png')

            clean_image_path = (GROUND_TRUTH_PATH+list_clean_images[im])
            clean_image = Image.open(clean_image_path)# /255.0
            clean_image = clean_image.convert('L')
            clean_image.save('curr_clean_image.png')

            clean_image = plt.imread('curr_clean_image.png')#[:,:,0]


            wat_batch, gt_batch = getPatches(deg_image,clean_image,mystride=128+64)

            batch_count = wat_batch.shape[0] // batch_size




            for b in (range(batch_count)):
                seed= range(b*batch_size, (b*batch_size) + batch_size)
                b_wat_batch = wat_batch[seed].reshape(batch_size,256,256,1)
                b_gt_batch = gt_batch[seed].reshape(batch_size,256,256,1)

                generated_images = generator.predict(b_wat_batch)


                valid = np.ones((b_gt_batch.shape[0],) + (16, 16, 1))
                fake = np.zeros((b_gt_batch.shape[0],) + (16, 16, 1))
    
                discriminator.trainable = True          
                d_loss_real, d_acc_real = discriminator.train_on_batch([b_gt_batch, b_wat_batch], valid)
                d_loss_fake, d_acc_fake = discriminator.train_on_batch([generated_images, b_wat_batch], fake)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)

                # Log discriminator loss to TensorBoard
                with discriminator_writer.as_default():
                    tf.summary.scalar('d_loss', d_loss, step=global_step)
                    tf.summary.scalar('d_accuracy', d_acc, step=global_step)

                discriminator.trainable = False
                g_loss = gan.train_on_batch([b_wat_batch], [valid, b_gt_batch])

                # Log generator loss to TensorBoard (g_loss is a list: [total_loss, mse_loss_disc_output, bce_loss_generator_output, acc_disc_output, acc_generator_output])
                with generator_writer.as_default():
                    tf.summary.scalar('g_total_loss', g_loss[0], step=global_step)
                    tf.summary.scalar('g_disc_output_loss', g_loss[1], step=global_step)
                    tf.summary.scalar('g_image_loss', g_loss[2], step=global_step)
                    tf.summary.scalar('g_disc_output_accuracy', g_loss[3], step=global_step) # This is the accuracy of the discriminator on generator output when generator tries to fool it.

                global_step += 1

                if b % 10 == 0:  # Print every 10 batches
                    print(
                        f"  Batch {b}/{batch_count} | D Loss: {d_loss:.4f} (Acc: {d_acc:.4f}) | G Loss: {g_loss[0]:.4f} (Disc Acc: {g_loss[3]:.4f})")

        epoch_weights_path = f'trained_weights/epoch_{e}'
        if not os.path.exists(epoch_weights_path):
            os.makedirs(epoch_weights_path)
        discriminator.save_weights(f'{epoch_weights_path}/discriminator.weights.h5')
        generator.save_weights(f'{epoch_weights_path}/generator.weights.h5')
        # if (e == 1 or e % 2 == 0):
        #     evaluate(generator,discriminator,e)
    
def predic(generator, epoch):
    if not os.path.exists('Results/epoch'+str(epoch)):
        os.makedirs('Results/epoch'+str(epoch))
    for i in range(0,31):
        watermarked_image_path = ('CLEAN/VALIDATION/DATA/'+ str(i+1) + '.png')
        test_image = plt.imread(watermarked_image_path)
        
        h =  ((test_image.shape [0] // 256) +1)*256 
        w =  ((test_image.shape [1] // 256 ) +1)*256
        
        test_padding=np.zeros((h,w))+1
        test_padding[:test_image.shape[0],:test_image.shape[1]]=test_image
        
        test_image_p=split2(test_padding.reshape(1,h,w,1),1,h,w)
        predicted_list=[]
        for l in range(test_image_p.shape[0]):
            predicted_list.append(generator.predict(test_image_p[l].reshape(1,256,256,1)))
        
        predicted_image = np.array(predicted_list)#.reshape()
        predicted_image=merge_image2(predicted_image,h,w)
        
        predicted_image=predicted_image[:test_image.shape[0],:test_image.shape[1]]
        predicted_image=predicted_image.reshape(predicted_image.shape[0],predicted_image.shape[1])
        predicted_image = (predicted_image[:,:])*255
        
        predicted_image =predicted_image.astype(np.uint8)
        imageio.imwrite('Results/epoch'+str(epoch)+'/predicted'+str(i+1)+'.png', predicted_image)




### if you want to evaluate each epoch:


# def  evaluate(generator,discriminator,epoch):
#     predic(generator,epoch)
#     avg_psnr=0
#     qo=0

#     for i in range (0,31):
        
#         test_image= plt.imread('CLEAN/VALIDATION/GT/'+ str(i+1) + '.png')

#         predicted_image= plt.imread('Results/epoch'+str(epoch)+'/predicted'+ str(i+1) + '.png')
#         avg_psnr= avg_psnr + psnr(test_image,predicted_image)
#         qo=qo+1
#     avg_psnr=avg_psnr/qo
#     print('psnr= ',avg_psnr)
#     if not os.path.exists('Results/epoch'+str(epoch)+'/weights'):
#         os.makedirs('Results/epoch'+str(epoch)+'/weights')
#     discriminator.save_weights("Results/epoch"+str(epoch)+"/weights/discriminator_weights.h5")
#     generator.save_weights("Results/epoch"+str(epoch)+"/weights/generator_weights.h5")


##################################

epo = 1

generator = generator_model(biggest_layer=1024)
discriminator = discriminator_model()


### to  load pretrained models  ################"" 
# epo = 39
#
# epoch_weights_path = f'trained_weights/epoch_{epo-1}'
# generator.load_weights(f"{epoch_weights_path}/generator.weights.h5")
# discriminator.load_weights(f"{epoch_weights_path}/discriminator.weights.h5")


###############################################

train_gan(generator,discriminator, ep_start =epo, epochs=100, batch_size=1)