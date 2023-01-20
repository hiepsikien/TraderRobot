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

    def run(self,gpu = True, epochs = 200):
       
        path_checkpoint = "../logs/model/model_lstm_checkpoint.h5"
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
                epochs= epochs,
                verbose=2,
                validation_data=self.dataset_val,  
                callbacks=[es_callback, modelckpt_callback],
            )
        
        self.saved_history = dict(self.history.history)

        with tf.device(processor):    
            self.pred_prob = self.model.predict(x=self.dataset_test)

        self.analyze_predict_by_cutoff()
        visualizer.visualize_efficiency_by_cutoff(self.efficiency,0,0.5)

        accuracy, coverage = self.filter_prediction_by_cutoff_for_binary(
            neg_cutoff=self.neg_cutoff,
            pos_cutoff=self.pos_cutoff
        )

        return accuracy, coverage
    

       


