
from platform import processor
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras import callbacks as kc
from keras.optimizers import Adam
from base_classifier import BaseClassifierModel

class DNNClassifer(BaseClassifierModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)

    def configure(self,hl,hu, input_dim, dropout = False, regularize = False, reg = l1(0.0005)):
        if not regularize:
            reg = None
        
        self.model = Sequential()

        self.model.add(Dense(hu,input_dim = input_dim, activity_regularizer=reg, activation="relu"))
        
        if dropout:
            self.model.add(Dropout(rate = self.dropout_rate,seed =self.seeds ))

        for layer in range(hl):
            self.model.add(Dense(hu,activation="relu",activity_regularizer=reg))
            if dropout:
                self.model.add(Dropout(rate=self.dropout_rate,seed=self.seeds))
        
        optimizer = Adam(learning_rate = 0.0001)

        self.model.add(Dense(1,activation="sigmoid"))
        
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"]
        )

    def prepare_data(self, data, cols, random_state = 1, shuffle = False):

        # Shuffle data
        if shuffle:
            shuffled = data.sample(frac=1,random_state=random_state)
        else:
            shuffled = data
        
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

    def run(self,gpu):
       
        path_checkpoint = "../data/model_dnn_checkpoint.h5"
        es_callback = kc.EarlyStopping(monitor="val_loss", min_delta=0, verbose=1, patience=3)

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
                x=self.x_train,
                y=self.y_train,
                epochs=self.epochs,
                verbose=2,
                validation_data=(self.x_val,self.y_val), 
                shuffle=True, 
                callbacks=[es_callback],
                class_weight=self.cw(self.y_train))

        self.saved_history = dict(self.history.history)
    
        with tf.device(processor):
            self.pred_prob = self.model.predict(x=self.x_test)
        
        accuracy, coverage = self.filter_prediction_by_cutoff(
            neg_cutoff=self.neg_cutoff,
            pos_cutoff=self.pos_cutoff)

        return accuracy, coverage