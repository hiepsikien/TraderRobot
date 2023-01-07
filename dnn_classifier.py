from itertools import count
from pickletools import optimize
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

class DNNModel(Sequential):

    def __init__(self,seed = 100, dropout_rate = 0.3, neg_cutoff = 0.45, pos_cutoff = 0.55, test_size = 0.3, validation_split = 0.2, epochs=50, optimizer = Adam(learning_rate=0.0001)) -> None:
        super().__init__()
        self.set_seeds(seed)
        self.set_optimizer(optimizer)
        self.set_cutoff(neg_cutoff = neg_cutoff, pos_cutoff =pos_cutoff)
        self.set_test_size(test_size=test_size)
        self.set_validation_split(validation_split)
        self.set_epochs(epochs)
        self.set_dropout_rate(dropout_rate)

    def set_dropout_rate(self,dropout_rate):
        self.dropout_rate = dropout_rate

    def set_epochs(self,epochs):
        self.epochs = epochs

    def set_test_size(self, test_size):
        self.test_size = test_size

    def set_validation_split(self, valiation_split):
        self.validation_split = valiation_split

    def set_cutoff(self,neg_cutoff, pos_cutoff):
        self.nt = neg_cutoff
        self.pt = pos_cutoff

    def set_seeds(self,seed):
        self.seeds = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def set_optimizer(self,optimizer):
        self.optimizer = optimizer

    def cw(self,data):
        c0, c1 = np.bincount(data)
        w0 = (1/c0) * (len(data))/2
        w1 = (1/c1) * (len(data))/2
        return {0:w0,1:w1}


    def configure(self,hl,hu, input_dim, dropout = False, regularize = False, reg = l1(0.0005)):
        if not regularize:
            reg = None
        
        self.add(Dense(hu,input_dim = input_dim, activity_regularizer=reg, activation="relu"))
        
        if dropout:
            self.add(Dropout(rate = self.dropout_rate,seed =self.seeds ))

        for layer in range(hl):
            self.add(Dense(hu,activation="relu",activity_regularizer=reg))
            if dropout:
                self.add(Dropout(rate=self.dropout_rate,seed=self.seeds))
        
        self.add(Dense(1,activation="sigmoid"))
        self.compile(loss="binary_crossentropy",optimizer=self.optimizer,metrics=["accuracy"])

    def run(self,data,cols):
       
        x_train, x_test, y_train, y_test = train_test_split(data[cols],data["dir"],test_size=self.test_size)
        
        self.fit(x=x_train,y=y_train,epochs=self.epochs,verbose=False,validation_split=self.validation_split, shuffle=False, class_weight=self.cw(y_train))
        
        pred_prob = self.predict(x=x_test)
        
        temp = np.where(pred_prob <self.nt,0,pred_prob)
        y_pred = np.where(temp >self.pt,1,temp)

        dfs = pd.DataFrame({"y_test":y_test,"y_pred":y_pred.flatten().tolist()})
        dfs["correct"] = dfs["y_test"] == dfs["y_pred"]

        correct_num = dfs["correct"].loc[dfs["correct"]==True].size
        pred_num = dfs["y_pred"].loc[(dfs["y_pred"] == 0) | (dfs["y_pred"] == 1) ].size
        all_num = dfs["correct"].size

        print("Correct = {}, Predict = {}, All = {} ".format(correct_num,pred_num,all_num))

        accuracy = correct_num / pred_num
        coverage = pred_num / all_num

        print("Accuracy Score: {}, Coverage Score: {}".format(round(accuracy,3), round(coverage,3)))
            
  