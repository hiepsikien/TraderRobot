
from tracemalloc import start
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras import callbacks as kc
from keras.optimizers import Adam
from visualizer import visualize_efficiency_by_cutoff

class DNNModel(Sequential):

    def __init__(self,seed = 100, dropout_rate = 0.3, neg_cutoff = 0.45, pos_cutoff = 0.55, train_size = 0.7, val_size =0.15, epochs=20, optimizer = Adam(learning_rate=0.0001)) -> None:
        kc.backend.clear_session()
        super().__init__()
        self.set_seeds(seed)
        self.set_optimizer(optimizer)
        self.set_cutoff(neg_cutoff = neg_cutoff, pos_cutoff =pos_cutoff)
        self.set_train_size(train_size)
        self.set_val_size(val_size)
        self.set_epochs(epochs)
        self.set_dropout_rate(dropout_rate)
        self.saved_history = None

    def set_dropout_rate(self,dropout_rate):
        self.dropout_rate = dropout_rate

    def set_epochs(self,epochs):
        self.epochs = epochs

    def set_val_size(self, val_size):
        self.val_size = val_size

    def set_train_size(self, train_size):
        self.train_size = train_size

    def set_cutoff(self,neg_cutoff, pos_cutoff):
        self.neg_cutoff = neg_cutoff
        self.pos_cutoff = pos_cutoff

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

    def visualize_loss(self):
        if self.saved_history is not None:
            loss = self.saved_history["loss"]
            val_loss = self.saved_history["val_loss"]
            epochs = range(len(loss))
            plt.figure(figsize=(8,6))
            plt.plot(epochs, loss, "b", label="Training loss")
            plt.plot(epochs, val_loss, "r", label="Validation loss")
            plt.title("Training and Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        else:
            print("No learning history")

    def visualize_efficiency_by_cutoff(self, min_delta = 0, max_delta = 0.5):
        visualize_efficiency_by_cutoff(data = self.efficiency,min_delta=min_delta,max_delta=max_delta)

    def visualize_accuracy(self):
        if self.saved_history is not None:
            accuracy = self.saved_history["accuracy"]
            val_accuracy = self.saved_history["val_accuracy"]
            epochs = range(len(accuracy))
            plt.figure(figsize=(8,6))
            plt.plot(epochs, accuracy, "b", label="Training accuracy")
            plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
            plt.title("Training and Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()
        else:
            print("No learning history")

    def split_data(self,data, cols,random_state = 1):

        # Shuffle data
        shuffled = data.sample(frac=1,random_state=random_state)

        #Calculate length
        data_len = len(data["dir"])
        train_len = int(data_len*self.train_size)
        val_len = int(data_len*self.val_size)
        test_len = data_len - train_len - val_len

        print("Train = {}, Val = {}, Test = {}, All = {}".format(train_len,val_len,test_len,data_len))

        #Split data to train + validation + test
        input = shuffled[cols]
        target = shuffled["dir"]

        self.x_train = input.head(train_len).values.tolist()
        self.y_train = target.head(train_len).tolist()

        self.x_val = input.iloc[train_len:train_len+val_len].values.tolist()
        self.y_val = target.iloc[train_len:train_len+val_len].tolist()

        self.x_test = input.tail(test_len).values.tolist()
        self.y_test = target.tail(test_len).tolist()

    def predict_with_cutoff(self,neg_cutoff,pos_cutoff): 
        temp = np.where(self.pred_prob < neg_cutoff,0,self.pred_prob)
        y_pred = np.where(temp > pos_cutoff,1,temp)

        dfs = pd.DataFrame({"y_test":self.y_test,"y_pred":y_pred.flatten().tolist()})
        dfs["correct"] = dfs["y_test"] == dfs["y_pred"]

        correct_num = dfs["correct"].loc[dfs["correct"]==True].size
        pred_num = dfs["y_pred"].loc[(dfs["y_pred"] == 0) | (dfs["y_pred"] == 1) ].size
        all_num = dfs["correct"].size

        print("Correct = {}, Predict = {}, All = {} ".format(correct_num,pred_num,all_num))

        accuracy = 0
        if pred_num != 0:
            accuracy = correct_num / pred_num
        
        coverage = pred_num / all_num

        print("Accuracy Score: {}, Coverage Score: {}".format(round(accuracy,3), round(coverage,3)))
        
        return accuracy, coverage

    def analyze_predict_by_cutoff(self):

        step = 0.001
        
        neg_cutoff = None
        pos_cutoff = None

        acc_list= []
        cov_list = []
        del_list = []
        
        for i in range(500):
            delta = i * step
            neg_cutoff = 0.5 - delta
            pos_cutoff = 0.5 + delta
            acc, cov = self.predict_with_cutoff(neg_cutoff=neg_cutoff,pos_cutoff=pos_cutoff)
            acc_list.append(acc)
            cov_list.append(cov)
            del_list.append(delta)

        self.efficiency = pd.DataFrame({"delta":del_list,"accuracy":acc_list,"coverage":cov_list})

    def run(self):
       
        path_checkpoint = "model_checkpoint.h5"
        es_callback = kc.EarlyStopping(monitor="val_loss", min_delta=0, verbose=1, patience=3)

        modelckpt_callback = kc.ModelCheckpoint(
            monitor="val_loss",
            filepath=path_checkpoint,
            verbose=1,
            save_weights_only=True,
            save_best_only=True)

        self.history = self.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=self.epochs,
            verbose=2,
            validation_data=(self.x_val,self.y_val), 
            shuffle=True, 
            callbacks=[es_callback],
            class_weight=self.cw(self.y_train))

        self.saved_history = dict(self.history.history)
        
        self.pred_prob = self.predict(x=self.x_test)
        
        accuracy, coverage = self.predict_with_cutoff(neg_cutoff=self.neg_cutoff,pos_cutoff=self.pos_cutoff)

        return accuracy, coverage