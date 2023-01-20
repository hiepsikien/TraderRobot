import pandas as pd
from feature_manager import FeatureManager
import utils
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import callbacks
from keras import metrics
from keras.regularizers import l1
from keras.optimizers import Adam
from classifier.base_classifier import BaseClassifier
import datetime
import utils
from sklearn.metrics import classification_report, confusion_matrix

METRICS = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'), 
      metrics.BinaryAccuracy(name='accuracy'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall'),
      metrics.AUC(name='auc'),
      metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

class MultiDNNClassifer(BaseClassifier):

    def __init__(self, *args, **kwargs) -> None:
        ''' Initialize
        '''
        super().__init__(*args,**kwargs)

    def configure(self,hu, input_dim, class_num, 
        output_bias = None, 
        loss="categorical_crossentropy", 
        dropout = False, 
        regularize = False, 
        reg = l1(0.0005),
        learning_rate = 0.0001):
        ''' 
        Create and configure model
            
        Params:
        - hu: number of cells for the biggest layer
        - input_dim: number of inputs for the model, equivalent to number of features in input data
        - class_num: number of output, equivalent to number of category
        
        Return: None
        '''

        self.params["hu"] = hu
        self.params["output_bias"] = output_bias
        self.params["loss"] = loss
        self.params["dropout"] = dropout
        if(dropout):
            self.params["dropout_rate"] = self.dropout_rate
        self.params["learning_rate"] = learning_rate

        if not regularize:
            reg = None

        self.model = Sequential()
        self.model.add(Dense(hu,input_dim = input_dim, activity_regularizer=reg, activation="relu",name="Dense1"))
        if dropout:
            self.model.add(Dropout(rate = self.dropout_rate,seed =self.seeds,name="Dropout1"))
        # self.model.add(Dense(int(hu/4),activity_regularizer=reg,activation="relu",name="Dense2"))
        # if dropout:
        #     self.model.add(Dropout(rate = self.dropout_rate,seed =self.seeds,name="Dropout2"))
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.model.add(Dense(units = class_num,activation="softmax", name="Output", bias_initializer=output_bias)) # type: ignore 
        
        self.model.compile(
            loss=loss,
            optimizer=Adam(learning_rate = learning_rate),
            metrics = METRICS
        )

    def run(self, fm:FeatureManager, gpu = False, set_class_weight = False,
        save_check_point = True, early_stopping= True,
        patience = 5, epochs = 200, batch_size = 10, file = None):
        '''  
        Fit model, store histories, evalute with test data, print report
        
        Params:
        - gpu: True if run with GPU, False if CPU
        - patience: number of epoch to try if no imporvement

        Returns: evalute result on test data
        '''

        self.params["gpu"] = gpu
        self.params["set_class_weight"] = set_class_weight
        self.params["save_check_point"] = save_check_point
        self.params["early_stopping"] = early_stopping
        if early_stopping:
            self.params["patience"] = patience
        self.params["epochs"] = epochs
        self.params["batch_size"] = batch_size 

        self.print_params(fm.params,file = file)

        callback_list = []

        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path_checkpoint = "../logs/model/model_multi_dnn_checkpoint_{}.h5".format(time)
        
        if early_stopping:
            es_callback = callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0, 
                verbose=1, 
                patience=patience)
            callback_list.append(es_callback)  

        if save_check_point:
            modelckpt_callback = callbacks.ModelCheckpoint(
                monitor="val_loss",
                filepath=path_checkpoint,
                verbose=1,
                save_weights_only=True,
                save_best_only=True)
            callback_list.append(modelckpt_callback)

        log_dir = "../../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callback_list.append(tensorboard_callback)

        processor = "/cpu:0"
        if (gpu):
            processor = "/gpu:0"

        if set_class_weight:
            class_weight=utils.calculate_weight(
                data = self.data_train,
                target_col=fm.target_col
            )
        else:
            class_weight=None

        with tf.device(processor):  # type: ignore 
            self.history = self.model.fit(
                x = self.x_train,
                y = self.y_train,
                epochs=epochs,
                verbose=2, # type: ignore
                batch_size=batch_size,
                validation_data=(self.x_val,self.y_val), 
                shuffle=True,
                class_weight=class_weight,
                callbacks=callback_list
            )

        self.saved_history = dict(self.history.history)  #type: ignore
    
        with tf.device(processor): # type: ignore 
            self.pred_prob = self.model.predict(x=self.x_test)
            self.pred_class = np.argmax(self.pred_prob,axis=-1)

            test_results = self.model.evaluate(
                self.x_test,
                self.y_test,
                batch_size = batch_size,
                return_dict = True
            )

        self.print_classification_report(file = file)
        self.print_confusion_matrix(file = file)

        return test_results

    def print_params(self,pre_params:dict,file=None):
        '''Print parameter of the classifier together with required extras
        '''
        params = pre_params | self.params
        print("\n=============",file = file)
        print("PARAMS:",file = file)
        for key in params.keys():
            print("{}:{}".format(key,params[key]),file = file)

    def print_classification_report(self,file=None):
        '''Print classification report
        '''
        y_pred_int = np.argmax(self.pred_prob, axis=1)
        y_test_int = np.argmax(self.y_test,axis = 1)    #type: ignore
        print("\n=============",file = file)
        print("CLASSIFICATION REPORT:",file = file)
        print(classification_report(y_test_int, y_pred_int),file = file)

    def print_confusion_matrix(self, file = None):
        '''Print confusion matrix'''
        print("\n=============",file = file)
        print("CONFUSION MATRIX:",file = file)
        con_matrix = confusion_matrix(np.argmax(self.y_test,axis=-1),self.pred_class) #type: ignore
        df = pd.DataFrame(con_matrix, columns = ["P-{}".format(i) for i in range(0, len(con_matrix))])
        df.loc[:,"Total"]= df.sum(axis = 1, numeric_only = True).astype(int)
        df.loc["Total"] = df.sum(axis = 0, numeric_only = True).astype(int)
        for i in range(0, len(con_matrix)):
            df["RP-{}".format(i)] = round(df["P-{}".format(i)]/df["Total"],3)
        print(df,file = file)
        
def loop_classifier(hu:int,fm: FeatureManager,laps:int, gpu:bool=False, 
    save_check_point:bool = True, early_stopping:bool = True, set_class_weight:bool = False,
    epochs:int = 200, patience:int = 5, batch_size:int = 24, write_to_file:bool = False):
    ''' 
    Run the classifier multiple time with loop calculate the average and std value of accuracy and loss

    Params:
    - fm: Feature Manager instance that store features and target values
    - target_col: name of target column
    - laps: number of lap to loop
    
    Return: None
    '''

    #Create a file
    report_file = None
    if write_to_file:
        filename = "../logs/report/{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        report_file = open(filename,'w')

    accuracy_list = []
    recall_list = []
    precision_list = []
    loss_list = []

    print_features_list(fm=fm,file=report_file)

    for i in range (0,laps):
        print("\n>>>>>> LAP {}\n".format(i+1), file=report_file)
        callbacks.backend.clear_session()
        classifier = MultiDNNClassifer()
        
        classifier.prepare_data(
            data = fm.df,
            cols = fm.cols,
            is_shuffle = True,
            y_to_categorical=True,
            random_state=i+1,
            target_col=fm.target_col,
            file = report_file
        )

        initial_bias = utils.init_imbalanced_bias(
            data=classifier.data_train,
            target_col=fm.target_col
        )

        classifier.configure(
            hu = hu, 
            dropout=True, 
            input_dim=len(fm.cols),
            output_bias=initial_bias,
            class_num=3,
        )

        processor = "/cpu:0"
        with tf.device(processor):                          # type: ignore 
            test_results = classifier.run(
                gpu = gpu,
                fm =fm,
                epochs=epochs, 
                patience=patience,
                early_stopping = early_stopping,
                save_check_point = save_check_point,
                set_class_weight = set_class_weight,
                batch_size = batch_size,
                file = report_file)

        # classifier.visualize_loss()
        # classifier.visualize_accuracy()

        loss_list.append(test_results["loss"])              #type: ignore
        accuracy_list.append(test_results["accuracy"])      #type: ignore
        precision_list.append(test_results["precision"])    #type: ignore
        recall_list.append(test_results["recall"])          #type: ignore

    results_dict = dict()
    results_dict["loss"] = loss_list
    results_dict["accuracy"] = accuracy_list
    results_dict["precision"] = precision_list
    results_dict["recall"] = recall_list

    print_test_summary(results_dict=results_dict,file=report_file) #type:_ignore
    
    if report_file is not None:
        report_file.close()

def print_features_list(fm: FeatureManager, file = None):
    print("\n=============",file = file)
    print("FEATURES (not including lag):",file = file)
    for i in range(int(len(fm.cols)/fm.params["lags"])):
        print("{},".format(fm.cols[i]),end=" ",file = file)
    print("\n",file=file)

def print_test_summary(results_dict:dict,file = None):
    print("\n>>>>>>",file = file)
    print("EVALUATION SUMMARY:",file = file)
    for key in results_dict.keys():
        temp = np.array(results_dict[key])
        print("{} mean: {}, std: {}".format(key, temp.mean(),temp.std()),file = file)
    print("\n",file = file)
