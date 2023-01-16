
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
from sklearn.metrics import classification_report
from random import randint

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

        self.model.add(Dense(int(hu/4),activity_regularizer=reg,activation="relu"))
        
        if dropout:
            self.model.add(Dropout(rate = self.dropout_rate,seed =self.seeds ))

        self.model.add(Dense(class_num,activation="softmax"))
        
        self.model.compile(
            loss=loss,
            optimizer=Adam(learning_rate = 0.0001),
            metrics=["accuracy"]
        )
        # self.model.summary()

    def prepare_data(self, data, cols, target_col, random_state = 1, shuffle = True,rebalance = True):

        # Resampling to make the imbalance training data
        self.data_for_classifier = data.copy()
        
        if rebalance:
            val_counts = self.data_for_classifier[target_col].value_counts().sort_index()
            weights = val_counts.max()/val_counts
            data_lst = []
            for i in val_counts.index:
                data_by_val = self.data_for_classifier[self.data_for_classifier["trade_signal"]==i].sample(replace=True,frac=weights[i],random_state=randint(0,100))
                data_lst.append(data_by_val)
            self.data_for_classifier = pd.concat(data_lst)

        # Shuffle data
        if shuffle:
            self.data_for_classifier = self.data_for_classifier.sample(frac=1,random_state=random_state)
        
        #Calculate length
        data_len = len(self.data_for_classifier)
        train_len = int(data_len*self.train_size)
        val_len = int(data_len*self.val_size)
        test_len = data_len - train_len - val_len

        print("Train = {}, Val = {}, Test = {}, All = {}".format(train_len,val_len,test_len,data_len))

        #Split data to train + validation + test
        input = self.data_for_classifier[cols]
        target = np_utils.to_categorical(self.data_for_classifier[target_col])

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

    def print_classification_report(self):
        y_pred_int = np.argmax(self.pred_prob, axis=1)
        y_test_int = np.argmax(self.y_test,axis = 1)
        print(classification_report(y_test_int, y_pred_int))