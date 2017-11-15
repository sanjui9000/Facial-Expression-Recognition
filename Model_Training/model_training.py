# some_file.py
import sys
sys.path.append('C:/Users/Sai/Desktop/Facial_Expression_Recognition/CNN_Model')

import numpy as np

# LambdaCallback is constructed with anonymous functions that will be called at the appropriate time. 
# Note that the callbacks expects positional arguments, as: on_epoch_begin/on_epoch_end,on_batch_begin/on_batch_end, on_train_begin/on_train_end
# Early stopping is used to stop  training when a monitored quantity has stopped improving.
from keras.callbacks import LambdaCallback, EarlyStopping

# Import model file
import myVGG

def main():
    model = myVGG.VGG_16()
    
    X_fname = 'C:/Users/Sai/Desktop/Facial_Expression_Recognition/Data_Generation/data/X_train_train.npy'
    y_fname = 'C:/Users/Sai/Desktop/Facial_Expression_Recognition/Data_Generation/data/y_train_train.npy'
    X_train = np.load(X_fname)
    y_train = np.load(y_fname)
    print(X_train.shape)
    print(y_train.shape)
   
    print("Training started")

    callbacks = []
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    batch_print_callback = LambdaCallback(on_batch_begin=lambda batch, logs: print(batch))
    epoch_print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print("epoch:", epoch))
    callbacks.append(earlystop_callback)
    callbacks.append(batch_print_callback)
    callbacks.append(epoch_print_callback)

    batch_size = 512
    model.fit(X_train, y_train, nb_epoch=400, \
            batch_size=batch_size, \
            validation_split=0.2, \
            shuffle=True, verbose=0, \
            callbacks=callbacks)

    model.save_weights('C:/Users/Sai/Desktop/Facial_Expression_Recognition/CNN_Model/my_model_weights.h5')
    scores = model.evaluate(X_train, y_train, verbose=0)
    print ("Train loss : %.3f" % scores[0])
    print ("Train accuracy : %.3f" % scores[1])
    print ("Training finished")

if __name__ == "__main__":
    main()
