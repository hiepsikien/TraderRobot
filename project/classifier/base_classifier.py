import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from visualizer import visualize_efficiency_by_cutoff
import tr_utils as tu
from keras.utils import np_utils
import numpy as np
from tr_printer import printb, print_labels_distribution


class BaseClassifier():

    def __init__(self,seed = 100, neg_cutoff = 0.45, pos_cutoff = 0.55) -> None:
        super().__init__()
        self.model = None
        self.set_seeds(seed)
        self.set_cutoff(neg_cutoff = neg_cutoff, pos_cutoff =pos_cutoff)
        self.saved_history = None
        self.params = dict()
    
    def set_cutoff(self,neg_cutoff, pos_cutoff):
        self.neg_cutoff = neg_cutoff
        self.pos_cutoff = pos_cutoff

    def set_seeds(self,seed):
        self.seeds = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

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

    def prepare_dataset(self, data, cols, target_col, train_size, val_size, sequence_len = 90, sequence_stride = 14, batch_size = 10, sampling_rate = 1, file = None):
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
        train_len = int(data_len*train_size)
        val_len = int(data_len*val_size)
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
        ''' Calculate y_test for dataset
        '''
        y_list = []
        window_len = (sequence_len-1)*sampling_rate + 1
        for end in range(window_len-1,len(y_test),sequence_stride):
            y_list.append(y_test[end])
        return y_list

    def prepare_data(self, data:pd.DataFrame, cols:list, target_col:str, train_size:float=0.7, val_size:float=0.15, random_state:int = 1, 
        shuffle_before_split:bool = True, categorical_label:bool = False, rebalance = None, cat_length:int = 1000,file = None):
        ''' 
        Make the data balance for all categories, shuffle and split for train, validation and test

        Params:
        - data: input data as dataframe
        - cols: list of features name
        - target_col: target column name
        - train_size: data train ratio, exp 0.7
        - val_size: data validation ratio, exp 0.15
        - random_state: a random state used for shuffle data
        - rebalance: "over": oversampling, "under": under sampling, "fix": fix the number of each cat to cat_length
        - cat_length: the number of each category if rebalance parameter is "fix"
        - categorical_label: if True, convert y_ data values to hot-bed, ex. 0: [1,0,0], 1:[0,1,0], 2:[0,0,1]
        
        Return: The list of data for training, validation and testing.
        '''

        self.params["random_state"] = random_state
        self.params["is_shuffle"] = shuffle_before_split
        self.params["categorical_label"] = categorical_label
        self.params["rebalance"] = rebalance
        
        if rebalance == "fix":
            self.params["cat_lentgh"] = cat_length

        # Shuffle data
        if shuffle_before_split:
            print("Shuffling the data...")
            shuffled = data.sample(frac=1,random_state=random_state)
        else:
            shuffled = data

        #Calculate length
        data_len = len(shuffled)
        train_len = int(data_len * train_size)
        val_len = int(data_len * val_size)
        test_len = data_len - train_len - val_len

        #Split data to train + validation + test
        print("Splitting the data...")
        self.data_train = shuffled.head(train_len).copy()
        self.data_val = shuffled.iloc[train_len:train_len+val_len].copy()
        self.data_test = shuffled.tail(test_len).copy()

        #Rebalance data for train and validation
        match rebalance:
            case "over":
                print("Rebalancing data with over-sampling")
                self.data_train = tu.over_sampling_rebalance(data=self.data_train,target_col=target_col)
                # self.data_val = tu.over_sampling_rebalance(data=self.data_val,target_col=target_col)
        
            case "under":
                print("Rebalancing data with under-sampling")
                self.data_train = tu.under_sampling_rebalance(data=self.data_train,target_col=target_col)
                # self.data_val = tu.under_sampling_rebalance(data=self.data_val,target_col=target_col)
       
            case "fix":
                print("Rebalancing data with fix category size {}".format(cat_length))
                self.data_train = tu.fix_sampling_rebalance(cat_length=cat_length,data=self.data_train,target_col=target_col)
                self.data_val = tu.fix_sampling_rebalance(cat_length=cat_length,data=self.data_val,target_col=target_col)
            
            case None:  
                pass

            case other:
                print("Failed to rebalance data due to wrong arguments")
            

        x_train = self.data_train[cols].values
        x_val = self.data_val[cols].values
        x_test = self.data_test[cols].values
        

        if categorical_label:
            y_train = np_utils.to_categorical(self.data_train[target_col].values,dtype="int64")
            y_val = np_utils.to_categorical(self.data_val[target_col].values,dtype="int64")
            y_test = np_utils.to_categorical(self.data_test[target_col].values,dtype="int64")
        else:
            y_train = self.data_train[target_col].values
            y_val = self.data_val[target_col].values
            y_test = self.data_test[target_col].values

        print("Data preparation completed.")
        printb("==========", file= file)
        printb("DATA:", file = file)
        printb("Data Train: {}, Validation: {}, Test: {}".
            format(len(self.data_train),len(self.data_val),len(self.data_test)),file = file)

        printb("\nTrain:", file = file)
        print_labels_distribution(self.data_train[target_col],file=file)

        printb("\nValidation:", file = file)
        print_labels_distribution(self.data_val[target_col],file=file)

        printb("\nTest:",file = file)
        print_labels_distribution(self.data_test[target_col],file=file)

        return [x_train,y_train,x_val,y_val,x_test,y_test]

    def run(self):
        pass

    def filter_prediction_by_cutoff_binary(self,neg_cutoff:float,pos_cutoff:float): 
        '''Filter test prediction with threshold, for binary classifier
        '''
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
            acc, cov = self.filter_prediction_by_cutoff_binary(
                neg_cutoff=neg_cutoff,
                pos_cutoff=pos_cutoff
            )
            acc_list.append(acc)
            cov_list.append(cov)
            del_list.append(delta)

        self.efficiency = pd.DataFrame({"delta":del_list,"accuracy":acc_list,"coverage":cov_list})





    