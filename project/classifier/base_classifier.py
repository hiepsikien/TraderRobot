import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from visualizer import visualize_efficiency_by_cutoff
import utils as tu
from keras.utils import np_utils

class BaseClassifier():

    def __init__(self,seed = 100, dropout_rate = 0.3, neg_cutoff = 0.45, pos_cutoff = 0.55, train_size = 0.7, val_size =0.15) -> None:
        super().__init__()
        self.model = None
        self.set_seeds(seed)
        self.set_cutoff(neg_cutoff = neg_cutoff, pos_cutoff =pos_cutoff)
        self.set_train_size(train_size)
        self.set_val_size(val_size)
        self.set_dropout_rate(dropout_rate)
        self.saved_history = None
    
    def set_dropout_rate(self,dropout_rate):
        self.dropout_rate = dropout_rate

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
        c0,c1 = np.bincount(data)
        w0 = 1/c0 * (c0+c1)/2
        w1 = 1/c1 * (c0+c1)/2
        return {0:w0,1:w1}

    def m_cw(self,data):
        counts = pd.DataFrame(data).value_counts()
        weights = 1/counts * counts.sum()/(len(counts))
        return {np.argmax(i,axis=-1):weights[i] for i in counts.index}

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

    def prepare_dataset(self, data, cols, target_col, sequence_len = 90, sequence_stride = 14, batch_size = 10, sampling_rate = 1):
        '''
        Prepare the dataset that required to feed by model such as LSTM

        Params:
        - data: input data as dataframe
        - cols: list of feature column name
        - target_col: target col name
        - sequence_len: the length of data windows
        - squence_stride: the step between one to the next window
        - batch_size: size of a batch
        - sampling_rate: the rate to pick up data

        Return: nothing. Store process data as attributes.
        '''

       #Calculate length
        data_len = len(data[target_col])
        train_len = int(data_len*self.train_size)
        val_len = int(data_len*self.val_size)
        test_len = data_len - train_len - val_len

        print("Train = {}, Val = {}, Test = {}, All = {}".format(train_len,val_len,test_len,data_len))

        #Split data to train + validation + test
        x_train = np.asarray(data[cols].head(train_len).values).astype(np.float32)
        y_train = data[target_col].head(train_len).values


        x_val = np.asarray(data[cols].iloc[train_len:train_len+val_len].values).astype(np.float32)
        y_val = data[target_col].iloc[train_len:train_len+val_len].values

        x_test = np.asarray(data[cols].tail(test_len).values).astype(np.float32)
        y_test = data[target_col].tail(test_len).values

        #Create the dataset for LSTM
        self.dataset_train = tf.keras.preprocessing.timeseries_dataset_from_array(
            x_train,
            y_train,
            sequence_length = sequence_len,
            sequence_stride= sequence_stride,
            batch_size = batch_size,
            sampling_rate = sampling_rate
        )

        self.dataset_val = tf.keras.preprocessing.timeseries_dataset_from_array(
            x_val,
            y_val,
            sequence_length = sequence_len,
            sequence_stride= sequence_stride,
            batch_size = batch_size,
            sampling_rate = sampling_rate
        )

        self.dataset_test = tf.keras.preprocessing.timeseries_dataset_from_array(
            x_test,
            y_test,
            sequence_length = sequence_len,
            sequence_stride= sequence_stride,
            batch_size = batch_size,
            sampling_rate = sampling_rate
        )

        self.y_test = self.prepare_y_test(
            y_test=y_test,
            sequence_len=sequence_len,
            sequence_stride=sequence_stride,
            sampling_rate=sampling_rate
        )

    def prepare_y_test(self,y_test,sequence_len,sequence_stride,sampling_rate):
        ''' Calculate y_test
        '''
        y_list = []
        window_len = (sequence_len-1)*sampling_rate + 1
        for end in range(window_len-1,len(y_test),sequence_stride):
            y_list.append(y_test[end])
        return y_list
    
    def prepare_data(self, data, cols, target_col, random_state = 1, shuffle = True, y_to_categorical = False, rebalance = None, cat_length = None):
        ''' 
        Make the data balance for all categories, shuffle and split for train, validation and test

        Params:
        - data: input data as dataframe
        - cols: list of features name
        - target_col: target column name
        - random_state: a random state used for shuffle data
        - rebalance: "over": oversampling, "under": under sampling, "fix": fix the number of each cat to cat_length
        - cat_length: the number of each category if rebalance parameter is "fix"
        - y_to_categorical: if True, convert y_ data values to hot-bed, ex. 0: [1,0,0], 1:[0,1,0], 2:[0,0,1]
        
        Return: nothing. The training, validation, test data is stored as objects attributes.
        '''
        # Shuffle data
        if shuffle:
            shuffled = data.sample(frac=1,random_state=random_state)
        else:
            shuffled = data

        #Calculate length
        data_len = len(shuffled)
        train_len = int(data_len*self.train_size)
        val_len = int(data_len*self.val_size)
        test_len = data_len - train_len - val_len

        #Split data to train + validation + test
        self.data_train = shuffled.head(train_len).copy()
        self.data_val = shuffled.iloc[train_len:train_len+val_len].copy()
        self.data_test = shuffled.tail(test_len).copy()

        #Rebalance data for train and validation
        match rebalance:
            case "over":
                self.data_train = tu.over_rebalance(data=self.data_train,target_col=target_col)
                self.data_val = tu.over_rebalance(data=self.data_val,target_col=target_col)
        
            case "under":
                self.data_train = tu.under_rebalance(data=self.data_train,target_col=target_col)
                self.data_val = tu.under_rebalance(data=self.data_val,target_col=target_col)
       
            case "fix":
                self.data_train = tu.fix_rebalance(cat_length=cat_length,data=self.data_train,target_col=target_col)
                self.data_val = tu.fix_rebalance(cat_length=cat_length,data=self.data_val,target_col=target_col)
            
            case None:  
                pass

            case other:
                print("Failed to rebalance data due to wrong arguments")
            

        self.x_train = self.data_train[cols].values
        self.x_val = self.data_val[cols].values
        self.x_test = self.data_test[cols].values
        

        if y_to_categorical:
            self.y_train = np_utils.to_categorical(self.data_train[target_col].values,dtype="int64")
            self.y_val = np_utils.to_categorical(self.data_val[target_col].values,dtype="int64")
            self.y_test = np_utils.to_categorical(self.data_test[target_col].values,dtype="int64")
        else:
            self.y_train = self.data_train[target_col].values
            self.y_val = self.data_val[target_col].values
            self.y_test = self.data_test[target_col].values

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

  