#!/usr/bin/env python
# coding: utf-8

# # Build a Generative Adversarial Neural Network 

# # 1-import dependencies and Data

# In[1]:


import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


os.listdir('fashion')


# In[3]:


data_train=pd.read_csv('fashion/fashion-mnist_train.csv')
data_test=pd.read_csv('fashion/fashion-mnist_test.csv')


# In[4]:


data_train


# In[5]:


pixels=data_train.values
pixels


# In[6]:


pixels.shape


# # 2-viz data and build pipeline

# In[8]:


import numpy as np


# In[9]:


#setup conection
dataiterator=tf.data.Dataset.from_tensor_slices(pixels).as_numpy_iterator()


# In[10]:


#gitting data out of the pipeline
dataiterator.next()[0]


# In[11]:


dataiterator.next().shape


# In[12]:


fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    img=dataiterator.next()
    image = img[1:].reshape(28, 28) 
    ax[idx].imshow(image, cmap='gray')  
    ax[idx].set_title(f'Label: {img[0]}')

plt.show()


# In[13]:


# scale images
def scale(img):
    return img/255 


# In[14]:


pixels=data_train.values
pixels


# In[15]:


pixels=pixels[:,1:]
pixels.shape


# In[16]:


original_size = pixels.size
original_size


# In[17]:


ds = tf.data.Dataset.from_tensor_slices(pixels)

def reshape_element(element):
    return tf.reshape(element, (28, 28, 1))

ds_reshaped = ds.map(reshape_element)

batch_size = 128  
ds = ds_reshaped.batch(batch_size)
ds=ds.map(scale)
#ds=ds.batch(128)


# In[18]:


ds.as_numpy_iterator().next().shape


# In[19]:


ds


# # 3-Build Neural Network

# # 3-1 import Modeling componenets

# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Reshape,LeakyReLU,Dropout,UpSampling2D


# # 3-2 Build Generator

# In[21]:


def build_generator():
    model=Sequential()
    model.add(Dense(7*7*128,input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
    
    model.add(UpSampling2D())
    model.add(Conv2D(128,5,padding='same'))
    model.add(LeakyReLU(0.2))
    
    model.add(UpSampling2D())
    model.add(Conv2D(128,5,padding='same'))
    model.add(LeakyReLU(0.2))
    
    #convolutional block1
    model.add(Conv2D(128,4,padding='same'))
    model.add(LeakyReLU(0.2))
    
    #convolutional block2
    model.add(Conv2D(128,4,padding='same'))
    model.add(LeakyReLU(0.2))
    

    #conv layer to get one channel
    model.add(Conv2D(1,4,padding='same',activation='sigmoid'))
    
    return model


# In[22]:


generator=build_generator()


# In[23]:


generator.summary()


# In[25]:


#to generate an img
img=generator.predict(np.random.randn(4,128,1))


# In[26]:


img.shape


# In[27]:


fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx,img in enumerate(img):
    #img=dataiterator.next()
    #image = img[1:].reshape(28, 28) 
    ax[idx].imshow(img, cmap='gray')  
    ax[idx].set_title(idx)

plt.show()


# # 3-3 Build Discriminator

# In[28]:


def build_discriminator():
    model=Sequential()
    
    #first conv block
    model.add(Conv2D(32,5,input_shape=(28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    #second conv block
    model.add(Conv2D(64,5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    #third conv block
    model.add(Conv2D(128,5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
     #fourth conv block
    model.add(Conv2D(256,5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    #flatten then pass to dense layer
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))
    return model


# In[29]:


discriminator=build_discriminator()


# In[30]:


discriminator.summary()


# In[31]:


img=generator.predict(np.random.randn(4,128,1))#4 dyal imgages


# In[32]:


img.shape


# In[33]:


discriminator.predict(img)


# In[34]:


imag=img[0]


# In[35]:


discriminator.predict(np.expand_dims(imag,0))


# # 4-Construct training loop

# # 4-1 setup losses and Optimizers

# In[36]:


#adam and binary for both
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


# In[37]:


g_opt=Adam(learning_rate=0.0001)
d_opt=Adam(learning_rate=0.00001)


# In[38]:


g_loss=BinaryCrossentropy()
d_loss=BinaryCrossentropy()


# # 4-2 build subclassed model

# In[39]:


#importing the base model class to subclass our training step
from tensorflow.keras.models import Model


# In[40]:


class FashionGAN(Model):
    
    def __init__(self,generator,discriminator,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.generator=generator
        self.discriminator=discriminator
        
    def compile(self,g_opt,d_opt,g_loss,d_loss,*args,**kwargs):
        #compile with base class
        super().compile(*args,**kwargs)
        self.g_opt=g_opt
        self.d_opt=d_opt
        self.g_loss=g_loss
        self.d_loss=d_loss
       
    def train_step(self,batch):
        #get data
        real_images=batch
        fake_images=self.generator(tf.random.normal((128,128,1)),training=False) 
       
        with tf.GradientTape() as d_tape:
            #pass real and fake img
            yhat_real=self.discriminator(real_images,training=True)
            yhat_fake=self.discriminator(fake_images,training=True)
            yhat_realfake=tf.concat([yhat_real,yhat_fake],axis=0)
            
            #create labels for real and fake img
            y_realfake=tf.concat([tf.zeros_like(yhat_real),tf.ones_like(yhat_fake)],axis=0)
            
            #add some noise to the outputs
            noise_real=0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake=-0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake+=tf.concat([noise_real,noise_fake],axis=0)
            
            #calculate the loss 
            total_d_loss=self.d_loss(y_realfake,yhat_realfake)
            
        #apply backpropagation
        dgrad=d_tape.gradient(total_d_loss,self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad,self.discriminator.trainable_variables))
        
        #train generator 
        with tf.GradientTape() as g_tape:
            #generate some new img
            gen_images=self.generator(tf.random.normal((128,128,1)),training=True)
            
            #create predicted labels
            predicted_labels=self.discriminator(gen_images,training=False)
            
            total_g_loss=self.g_loss(tf.zeros_like(predicted_labels),predicted_labels)
        
        #apply backpropa
        ggrad=g_tape.gradient(total_g_loss,self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad,self.generator.trainable_variables))
        
        
        return {"batch":batch,"d_loss" :total_d_loss,"g_loss":total_g_loss}


# In[41]:


#create instance of subclass model
fash=FashionGAN(generator,discriminator)


# In[42]:


#compile model
fash.compile(g_opt,d_opt,g_loss,d_loss)


# # 4-3 Build Callback

# In[43]:


import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback


# In[44]:


class ModelMonitor(Callback):
    def __init__(self,num_img=3,latent_dim=128):
        self.num_img=num_img
        self.latent_dim=latent_dim
    
    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim, 1))
        generated_img = self.model.generator(random_latent_vectors)
        generated_img *= 255
        generated_img = generated_img.numpy()

        for i in range(self.num_img):
            img = array_to_img(generated_img[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))


# In[45]:


import tensorflow as tf

# Generate random values from a uniform distribution between 0 and 1
uniform_values = tf.random.uniform(shape=(3, 3), minval=0, maxval=1)
uniform_values

tf.random.normal:

This function generates random values from a normal (Gaussian) distribution.
The normal distribution is characterized by a bell-shaped curve, and values near the mean are more likely to be sampled than values far from the mean.
It takes a shape parameter, specifying the shape of the output tensor, and you can also specify the mean and stddev parameters to define the mean and standard deviation of the distribution.
# In[46]:


import tensorflow as tf

# Generate random values from a normal distribution with mean 0 and standard deviation 1
normal_values = tf.random.normal(shape=(3, 3), mean=0, stddev=1)
normal_values


# # 4-3 Train

# In[47]:


ds.as_numpy_iterator().next().shape


# In[48]:


model_monitor = ModelMonitor()
hist=fash.fit(ds,epochs=1000,callbacks=[model_monitor],
    verbose=1)


# In[ ]:


plt.suptitle('Loss')
plt.plot(hist.history['d_loss'],label='d_loss')
plt.plot(hist.history['g_loss'],label='g_loss')
plt.legend()
plt.show()


# # 5-test our genetenerator

# # 5-1 genetare images

# In[ ]:


img=generator.predict(tf.random.normal((16,128,1)))
img


# In[ ]:


fig,ax=plt.subplots(ncols=4,nrows=4,figsize=(10,10))
for r in range(4):
    for c in range(4)
    ax[r][c].imshow(imgs[(r*c)-1])#maghaytl3 walo khs mn lahsen nht 2000 epoch


# # 5-2 save model

# In[ ]:


generator.save('fashion_gen.h5')
discriminator.save('fashion_dis.h5')

