
# Imports
import keras
import pickle 
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from keras import layers
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D, Lambda
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras import backend as K
import time
from gen_batches import gen_batches, gen_batches_validation, gen_batches_test 
from keras.losses import mse, binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


# Settings/hyperparameters
K.clear_session()
n_epochs = 50
batch_size = 16
learning_rate = 0.001
decay_rate = 0.05
n_train = 5776
n_val = 1000
n_test = 500
out_name = 'Variational_autoencoder'+'.pickle'
print(out_name)


# VAE encoder network
input_img = Input(shape=(48,48,48,4),name="Init_Input") 
x = layers.Conv3D(16, (3, 3, 3) , padding="same", activation='relu',name='E_Conv_1')(input_img)
x = layers.Conv3D(32, (3, 3, 3), padding="same", activation='relu',name='E_Conv_2')(x)
x = layers.MaxPooling3D((2,2,2))(x)
x = layers.Conv3D(48, (3, 3, 3), padding="same", activation='relu',name='E_Conv_3')(x)
x = layers.Conv3D(64, (3, 3, 3), padding="same", activation='relu',name='E_Conv_4')(x)
x = layers.MaxPooling3D((2,2,2))(x)
x = layers.Conv3D(96, (3, 3, 3), padding="same", activation='relu',name='E_Conv_5')(x)
x = layers.MaxPooling3D((2,2,2))(x)
x = layers.Conv3D(128, (3, 3, 3), padding="same", activation='relu',name='E_Conv_6')(x)
shape_before_flattening = K.int_shape(x)

# Latent space sampling function
latent_dim = 128
z_mean = layers.Dense(latent_dim,name='V_Mean')(x)
z_log_var = layers.Dense(latent_dim,name='V_Sig')(x)

    
def sampling(args):
    x, z_mean, z_log_var = args
    shape_before_flattening = K.int_shape(x)
    epsilon = K.random_normal(shape=(shape_before_flattening[1:]), mean=0., stddev=1.)
    return  z_mean + K.exp(z_log_var/2) * epsilon
    
z = layers.Lambda(sampling,name='V_Var')([x, z_mean, z_log_var])


# Decoder network 
decoder_input = layers.Input(shape=(shape_before_flattening[1:]),name='D_Input')
y = layers.Conv3DTranspose(96, 3, padding='same',activation='relu',name='D_Conv1')(decoder_input)
y = layers.Conv3D(64, 3,padding='same',activation='relu',name='D_Conv2')(y)
y = layers.UpSampling3D((2,2,2))(y)
y = layers.Conv3D(48, 3,padding='same',activation='relu',name='D_Conv3')(y)
y = layers.Conv3D(32, 3,padding='same',activation='relu',name='D_Conv4')(y)
y = layers.UpSampling3D((2,2,2))(y)
y = layers.Conv3D(16, 3,padding='same',activation='relu',name='D_Conv5')(y)
y = layers.UpSampling3D((2,2,2))(y)
y = layers.Conv3D(4, 3,padding='same',activation='relu',name='D_Conv6')(y)


#Instantiate the encoder, decoder
encoder = Model(input_img, [z_mean, z_log_var, z])
print(encoder.summary())
decoder = Model(decoder_input, y)
print(decoder.summary())

# Instantiate the VAE model
y = decoder(encoder(input_img)[2]) # Note that encoder provides 3 outputs
vae = Model(inputs=input_img, outputs=y) 
print(vae.summary())


# Define VAE loss function 
def vae_loss(input_img, y):
    reconstruction_loss = K.sum(K.square(input_img-y))
    kl_loss = - 0.5* K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)) 
    vae_loss = (reconstruction_loss + kl_loss)/batch_size
    return vae_loss


# Compile
vae.compile(optimizer='adam',loss=vae_loss, metrics=['accuracy'])


# Define callbacks
checkpoint = ModelCheckpoint("opt_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
earlystopping = EarlyStopping(patience=10)
tensorboard = TensorBoard(log_dir='./logs',update_freq='epoch')


#Fit the model
start = time.time()
steps = n_train // batch_size
val_steps = n_val // batch_size
history = vae.fit_generator(gen_batches(batch_size),
                                shuffle=True,
                                epochs=n_epochs,
                                steps_per_epoch= steps,
                                callbacks = [checkpoint, earlystopping, tensorboard],
                                validation_data = gen_batches_validation(batch_size),
                                validation_steps = val_steps)    
end = time.time()
print('Time Elapsed in minutes:',(end-start)/60)


# Save model and model weights 
vae.save_weights("vae_weights.h5")
encoder.save_weights("encoder_weights.h5")
decoder.save_weights("decoder_weights.h5")
print("Saved weights to disk")

# Save the model architecture
with open('vae_architecture.json', 'w') as f:
    f.write(vae.to_json())
    
with open('encoder_architecture.json', 'w') as f:
    f.write(encoder.to_json())
    
with open('decoder_architecture.json', 'w') as f:
    f.write(decoder.to_json())
    

# Display training loss, validation loss, epoch times
history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(train_loss)+1)

# Plot loss statistics 
plt.figure(0)
plt.plot(epochs, train_loss, 'r', label ='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss v/s epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss.png')

# Plot accuracy statistics 
plt.figure(1)
plt.plot(epochs, train_acc, 'r', label ='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training & validation accuracy v/s epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Accuracy.png')

# Calculating test scores
test_steps = n_test//batch_size
test_mse_score, test_acc_score = vae.evaluate_generator(gen_batches(batch_size),steps=test_steps, verbose=0)
test_mse= round(test_mse_score, 2)
test_acc= round(test_acc_score, 2)
test_MSE=' Test MSE:' + str(test_mse)
test_ACC=' Test acc:' + str(test_acc)
print (test_MSE, test_ACC)

data = {'train_loss':train_loss, 'val_loss':val_loss,'train_acc':train_acc, 'val_acc':val_acc, 'test_loss':test_mse, 'test_acc': test_acc}
pickle_out = open(out_name,"wb") # Save data in pickle file
pickle.dump(data, pickle_out)
pickle_out.close()

