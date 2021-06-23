import tensorflow as tf
import keras
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.transform import resize
import segmentation_models as sm
import keras.backend as K
from random import shuffle
from random import shuffle
from imgaug import augmenters as iaa

# custom loss
import custom_loss as custom
import loss_functions as lossF


######  Global Variable  #######
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# DATA_PATH = 'TODO_DATA_PATH'
# CHKP_PATH = 'TODO_CHKP_PATH'
# SAVE_PATH = 'TODO_SAVE_PATH'
LR = 1e-3
batch_size = 32
epoch = 30

# example: custom.[loss], sm.losses.[loss], lossF.[loss]
custom_loss = custom.TverskyLoss()
loss_name = 'Tver_lesion'

def get_model(img_shape=[None, None]):
    sm.set_framework('keras')
    
    model_seg = sm.Unet(input_shape=img_shape + [5],
                       classes = 1,
                       activation='sigmoid',
                       encoder_weights=None,
                       backbone_name='resnet34',
                       decoder_block_type='transpose',
                       )
    
    inp_all = keras.layers.Input(shape=img_shape + [5], name='input_seg')

    predictions = model_seg(inp_all)
    model = keras.Model(inputs=[inp_all], outputs=predictions)
    
    return model, model_seg

def generate_ct(num_set, minibatch, isVal, SAVE_NAME=None, DIR_DATA=None):
    while 1:
        cts = np.zeros((minibatch, 512, 512, 5))
        segs = np.zeros((minibatch, 512, 512, 1))
        
        # mini-batch counter
        for i in range(0, minibatch):
            
            num = random.sample(num_set, 1)
            
            ct, seg = generate_slice_randi(num, DIR_DATA=DIR_DATA)
            
            cts[i, :, :, :] = np.copy(ct)
            segs[i, :, :, :] = np.copy(seg)

            # Clip        
            cts = np.clip(cts, -200, 200)
            segs = segs.astype('uint8') # imgaug requires int type segs.
            
        ## isval
        if isVal is False:
            seq = iaa.Sequential([
                iaa.Sometimes(0.5,
                             # iaa.ElasticTransformation(alpha=90, sigma=9),
                             iaa.Rotate((-15, 15))
                             ),
                iaa.CropAndPad(percent=(-0.2, 0), pad_mode=['edge'], keep_size=True),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Resize(size=320, interpolation='cubic')
                ])
            
            cts, segs = seq(images=cts, segmentation_maps=segs)
            segs = segs.astype('float64')
            
        yield cts, segs
        
def generate_slice_randi(num, nout=3, DIR_DATA=None, SLIDE=False):
      
    data_vol = np.load(os.path.join(DIR_DATA, 'volume-%d.npy' % num[0]), mmap_mode='r+')
    data_seg = np.load(os.path.join(DIR_DATA, 'segmentation-%d.npy' % num[0]), mmap_mode='r+')

    if SLIDE:
            slide = SLIDE
    else:
        slide = random.randint(2, data_seg.shape[-1]-3)

        sliced_vol = data_vol[..., slide-2:slide+3] # 2.5D shape slides
        sliced_ct = sliced_vol
        
        sliced_seg = data_seg[...,slide]

        sliced_seg = sliced_seg[...,np.newaxis]
        sliced_seg = keras.utils.to_categorical(sliced_seg, num_classes=nout) # seg is (H, W, C=3) background, liver, lesion
        
        lesion_sliced_seg = np.zeros((512, 512, 1), dtype='float')
        lesion_sliced_seg[...,0] = sliced_seg[...,1] # only liver area
    
        return sliced_ct, lesion_sliced_seg
    
def main():
    # Train : Validation : Test = 6 : 2 : 2
    train_set = num_set[:95]
    val_set = num_set[95:110]
    test_set = num_set[110:]
    
    model, _ = get_model()

    model.compile(optimizer = keras.optimizers.Adam(LR),
                     loss = custom_loss,
                     metrics=[sm.metrics.FScore(per_image=False, name='Dice')])

    def scheduler(epoch, lr):
        return lr * (0.9 ** epoch)

    lrScheduler= keras.callbacks.LearningRateScheduler(scheduler)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    checkPoint = keras.callbacks.ModelCheckpoint(filepath=CHKP_PATH+loss_name+'.ckpt', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    lrOnPlateau = keras.callbacks.ReduceLROnPlateau(moniter='val_loss', factor=0.5, patience=5) 
    
    hist = model.fit_generator(
            generate_ct(train_set, batch_size, isVal=False, DIR_DATA= DATA_PATH),
            steps_per_epoch=(train_num//batch_size),
            epochs=epoch,
            validation_data=generate_ct(val_set, batch_size, isVal=True, DIR_DATA = DATA_PATH),
            validation_steps=(val_num//batch_size),
            callbacks=[lrScheduler, lrOnPlateau, checkPoint, earlyStopping],
            )
    
    np.save(SAVE_PATH+loss_name+'.npy', hist.history)


if __name__ == "__main__":
    main()
