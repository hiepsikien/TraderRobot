import numpy as np
import tensorflow as tf
from base_classifier import BaseClassifier
from keras.layers import Input, Dense, LSTM
from keras import Model, optimizers, callbacks
import visualizer

class LSTMClassifier(BaseClassifier):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)

    def configure(self, hu, shape):

        inputs = Input(shape=shape)
        lstm_out = LSTM(hu,recurrent_dropout=0.2)(inputs)
        outputs = Dense(1,activation="sigmoid")(lstm_out)

        self.model = Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics="accuracy")
        self.model.summary()

    def prepare_data(self, data, cols, sequence_len = 90, sequence_stride = 14, batch_size = 10, sampling_rate = 1):

       #Calculate length
        data_len = len(data["dir"])
        train_len = int(data_len*self.train_size)
        val_len = int(data_len*self.val_size)
        test_len = data_len - train_len - val_len

        print("Train = {}, Val = {}, Test = {}, All = {}".format(train_len,val_len,test_len,data_len))

        #Split data to train + validation + test
        x_train = np.asarray(data[cols].head(train_len).values).astype(np.float32)
        y_train = data["dir"].head(train_len).values


        x_val = np.asarray(data[cols].iloc[train_len:train_len+val_len].values).astype(np.float32)
        y_val = data["dir"].iloc[train_len:train_len+val_len].values

        x_test = np.asarray(data[cols].tail(test_len).values).astype(np.float32)
        y_test = data["dir"].tail(test_len).values

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
        y_list = []
        window_len = (sequence_len-1)*sampling_rate + 1
        for end in range(window_len-1,len(y_test),sequence_stride):
            y_list.append(y_test[end])
        return y_list

    def run(self,gpu):
       
        path_checkpoint = "../model_lstm_checkpoint.h5"
        es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, verbose=1, patience=3)

        modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath=path_checkpoint,
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
        )

        processor = "/cpu:0"

        if (gpu):
            processor = "/gpu:0"

        with tf.device(processor):
            self.history = self.model.fit(
                self.dataset_train,
                epochs= self.epochs,
                verbose=2,
                validation_data=self.dataset_val,  
                callbacks=[es_callback, modelckpt_callback],
            )
        
        self.saved_history = dict(self.history.history)

        with tf.device(processor):    
            self.pred_prob = self.model.predict(x=self.dataset_test)

        self.analyze_predict_by_cutoff()
        visualizer.visualize_efficiency_by_cutoff(self.efficiency,0,0.5)

        accuracy, coverage = self.filter_prediction_by_cutoff(
            neg_cutoff=self.neg_cutoff,
            pos_cutoff=self.pos_cutoff
        )

        return accuracy, coverage
    

       


