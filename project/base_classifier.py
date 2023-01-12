
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from visualizer import visualize_efficiency_by_cutoff

class BaseClassifierModel():

    def __init__(self,seed = 100, dropout_rate = 0.3, neg_cutoff = 0.45, pos_cutoff = 0.55, train_size = 0.7, val_size =0.15, epochs=20) -> None:
        super().__init__()
        self.model = None
        self.set_seeds(seed)
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

    def cw(self,data):
        c0, c1 = np.bincount(data)
        w0 = (1/c0) * (len(data))/2
        w1 = (1/c1) * (len(data))/2
        return {0:w0,1:w1}

    def configure(self):
        pass

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

    def prepare_data(self):
        pass
    
    def run(self):
        pass

    def filter_prediction_by_cutoff(self,neg_cutoff,pos_cutoff): 
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
            acc, cov = self.filter_prediction_by_cutoff(neg_cutoff=neg_cutoff,pos_cutoff=pos_cutoff)
            acc_list.append(acc)
            cov_list.append(cov)
            del_list.append(delta)

        self.efficiency = pd.DataFrame({"delta":del_list,"accuracy":acc_list,"coverage":cov_list})

  