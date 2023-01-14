
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import losses
from keras.regularizers import l1, l2
from keras import callbacks as kc
from keras.optimizers import Adam
from base_classifier import BaseClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

class MultiClassDNNClassifer(BaseClassifier):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)

    def configure(self,hu, input_dim, class_num, loss="categorical_crossentropy", dropout = False, regularize = False, reg = l1(0.0005)):
        if not regularize:
            reg = None
        
        self.model = Sequential()

        self.model.add(Dense(hu,input_dim = input_dim, activity_regularizer=reg, activation="relu"))
        
        if dropout:
            self.model.add(Dropout(rate = self.dropout_rate,seed =self.seeds ))

        self.model.add(Dense(int(hu/4), activation="relu"))
        
        if dropout:
            self.model.add(Dropout(rate = self.dropout_rate,seed =self.seeds ))

        self.model.add(Dense(class_num,activation="softmax"))
        
        self.model.compile(
            loss=loss,
            optimizer=Adam(learning_rate = 0.0001),
            metrics=["accuracy"]
        )
        self.model.summary()

    def prepare_data(self, data, cols, target_col = "dir", random_state = 1, shuffle = False):

        # Shuffle data
        if shuffle:
            shuffled = data.sample(frac=1,random_state=random_state)
        else:
            shuffled = data
        
        #Calculate length
        data_len = len(data[target_col])
        train_len = int(data_len*self.train_size)
        val_len = int(data_len*self.val_size)
        test_len = data_len - train_len - val_len

        print("Train = {}, Val = {}, Test = {}, All = {}".format(train_len,val_len,test_len,data_len))

        #Split data to train + validation + test
        input = shuffled[cols]
        target = np_utils.to_categorical(shuffled[target_col])

        self.x_train = input.head(train_len).values
        self.y_train = target[0:train_len]

        self.x_val = input.iloc[train_len:train_len+val_len].values
        self.y_val = target[train_len:train_len+val_len]

        self.x_test = input.tail(test_len).values
        self.y_test = target[train_len+val_len:data_len]

    def run(self,gpu = False, patience = 5, epochs = 100):
       
        path_checkpoint = "../data/model_dnn_checkpoint.h5"
        es_callback = kc.EarlyStopping(
            monitor="val_loss",
            min_delta=0, 
            verbose=1, 
            patience=patience)

        modelckpt_callback = kc.ModelCheckpoint(
            monitor="val_loss",
            filepath=path_checkpoint,
            verbose=1,
            save_weights_only=True,
            save_best_only=True)

        processor = "/cpu:0"

        if (gpu):
            processor = "/gpu:0"
        
        with tf.device(processor):
            self.history = self.model.fit(
                x = self.x_train,
                y = self.y_train,
                epochs=epochs,
                verbose=2,
                validation_data=(self.x_val,self.y_val), 
                shuffle=True,
                class_weight=self.m_cw(self.y_train),
                callbacks=[es_callback, modelckpt_callback]
            )

        self.saved_history = dict(self.history.history)
    
        with tf.device(processor):
            self.pred_prob = self.model.predict(x=self.x_test)
            self.pred_class = np.argmax(self.pred_prob,axis=-1)


        # accuracy, coverage = self.filter_prediction_by_cutoff(
        #     neg_cutoff=self.neg_cutoff,
        #     pos_cutoff=self.pos_cutoff)

        # return accuracy, coverage