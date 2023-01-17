import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import callbacks
from keras.regularizers import l1, l2
from keras import callbacks as kc
from keras.optimizers import Adam
from classifier.base_classifier import BaseClassifier
from sklearn.metrics import classification_report
from random import randint
import datetime

class MultiClassDNNClassifer(BaseClassifier):

    def __init__(self, *args, **kwargs) -> None:
        ''' Initialize
        '''
        super().__init__(*args,**kwargs)

    def configure(self,hu, input_dim, class_num, loss="categorical_crossentropy", dropout = False, regularize = False, reg = l1(0.0005)):
        ''' 
        Create and configure model
            
        Params:
        - hu: number of cells for the biggest layer
        - input_dim: number of inputs for the model, equivalent to number of features in input data
        - class_num: number of output, equivalent to number of category
        
        Return: None
        '''
        if not regularize:
            reg = None
        
        self.model = Sequential()
        self.model.add(Dense(hu,input_dim = input_dim, activity_regularizer=reg, activation="relu",name="Dense1"))
        if dropout:
            self.model.add(Dropout(rate = self.dropout_rate,seed =self.seeds,name="Dropout1"))
        self.model.add(Dense(int(hu/4),activity_regularizer=reg,activation="relu",name="Dense2"))
        if dropout:
            self.model.add(Dropout(rate = self.dropout_rate,seed =self.seeds,name="Dropout2"))
        self.model.add(Dense(class_num,activation="softmax",name="Output"))
        self.model.compile(
            loss=loss,
            optimizer=Adam(learning_rate = 0.0001),
            metrics=["accuracy"]
        )
        # self.model.summary()

    def run(self,gpu = False, patience = 5, epochs = 200, batch_size = 10):
        '''  
        Fit model, store histories, evalute with test data, print report
        
        Params:
        - gpu: True if run with GPU, False if CPU
        - patience: number of epoch to try if no imporvement

        Return: None
        '''
        path_checkpoint = "../logs/model/model_multi_dnn_checkpoint.h5"
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

        log_dir = "../../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        processor = "/cpu:0"

        if (gpu):
            processor = "/gpu:0"

        with tf.device(processor):
            self.history = self.model.fit(
                x = self.x_train,
                y = self.y_train,
                epochs=epochs,
                verbose=2,
                batch_size=batch_size,
                validation_data=(self.x_val,self.y_val), 
                shuffle=True,
                class_weight=self.m_cw(self.y_train),
                callbacks=[es_callback, modelckpt_callback,tensorboard_callback]
            )

        self.saved_history = dict(self.history.history)
    
        with tf.device(processor):
            self.pred_prob = self.model.predict(x=self.x_test)
            self.pred_class = np.argmax(self.pred_prob,axis=-1)

        self.print_classification_report()
        # accuracy, coverage = self.filter_prediction_by_cutoff(
        #     neg_cutoff=self.neg_cutoff,
        #     pos_cutoff=self.pos_cutoff)

        # return accuracy, coverage

    def print_classification_report(self):
        y_pred_int = np.argmax(self.pred_prob, axis=1)
        y_test_int = np.argmax(self.y_test,axis = 1)
        print(classification_report(y_test_int, y_pred_int))

def loop_classifier(data, cols,target_col,laps, gpu=False, epochs = 100, patience = 5, batch_size = 10):
    ''' 
    Run the classifier multiple time with loop calculate the average and std value of accuracy and loss

    Params:
    - data: the data as dataframe
    - cols: list of feature columns name
    - target_col: name of target column
    - laps: number of lap to loop
    
    Return: None
    '''
    acc_list = []
    loss_list = []

    for i in range (0,laps):
        print("\n======= Lap {} =======".format(i+1))
        callbacks.backend.clear_session()
        classifier = MultiClassDNNClassifer()
        classifier.configure(
            hu = 100, 
            dropout=True, 
            input_dim=len(cols),
            class_num=3
        )
        
        classifier.prepare_data(
            data = data,
            cols = cols,
            shuffle = True,
            y_to_categorical=True,
            random_state=i+1,
            target_col=target_col
        )

        print("y_train value counts:")
        print(pd.DataFrame(classifier.y_train).value_counts())

        print("y_val value counts:")
        print(pd.DataFrame(classifier.y_val).value_counts())

        print("y_test value counts:")
        print(pd.DataFrame(classifier.y_test).value_counts())

        processor = "/cpu:0"
        with tf.device(processor):
            classifier.run(gpu,epochs=epochs, patience=patience, batch_size = batch_size)

        classifier.visualize_loss()
        classifier.visualize_accuracy()

        test_results = classifier.model.evaluate(
            classifier.x_test,
            classifier.y_test,
            batch_size = batch_size
        )

        acc_list.append(test_results[1])
        loss_list.append(test_results[0])

    acc_arr = np.array(acc_list)
    loss_arr = np.array(loss_list)

    print("\n======")
    print("Accuracy mean: {}, std: {}".format(acc_arr.mean(),acc_arr.std()))
    print("Loss mean: {}, std: {}".format(loss_arr.mean(),loss_arr.std()))
    print("======")