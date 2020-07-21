#Utilização de GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  #-1, não usa a GPU

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())  #Exibe a CPU e se existir, exibe tambem a GPU

from keras import backend as back
back.tensorflow_backend._get_available_gpus() #Exibe somente as GPU

