from feature_manager import FeatureManager
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import callbacks, metrics
from keras.utils import np_utils
from keras.regularizers import l1
from keras.optimizers import Adam
from classifier.base_classifier import BaseClassifier
import datetime
from tr_utils import calculate_weight, init_imbalanced_bias
import tr_utils
from tr_printer import print_labels_distribution, printb
import tr_printer as trp

METRICS = [
    metrics.TruePositives(name='tp'),
    metrics.FalsePositives(name='fp'),
    metrics.TrueNegatives(name='tn'),
    metrics.FalseNegatives(name='fn'), 
    metrics.CategoricalAccuracy(name='accuracy'),
    metrics.Precision(name='precision'),
    metrics.Precision(name='precision-0.55',thresholds=0.55),  
    metrics.Precision(name='precision-0.60',thresholds=0.6),  
    metrics.Precision(name='precision-0.65',thresholds=0.65),
    metrics.Precision(name='precision-0.70',thresholds=0.70),
    metrics.Precision(name='precision-0.75',thresholds=0.75),
    metrics.Precision(name='precision-0.80',thresholds=0.80),
    metrics.Precision(name='precision-0.85',thresholds=0.85),
    metrics.Precision(name='precision-0.90',thresholds=0.90),
    metrics.Precision(name='precision-0.95',thresholds=0.95),   
    metrics.Recall(name='recall'),
    metrics.Recall(name='recall-0.55',thresholds=0.55),
    metrics.Recall(name='recall-0.60',thresholds=0.60),
    metrics.Recall(name='recall-0.65',thresholds=0.65),
    metrics.Recall(name='recall-0.70',thresholds=0.70),
    metrics.Recall(name='recall-0.75',thresholds=0.75),
    metrics.Recall(name='recall-0.80',thresholds=0.80),
    metrics.Recall(name='recall-0.85',thresholds=0.85),
    metrics.Recall(name='recall-0.90',thresholds=0.90),
    metrics.Recall(name='recall-0.95',thresholds=0.95),
]

class MultiDNNClassifer(BaseClassifier):

    def __init__(self, *args, **kwargs) -> None:
        ''' Initialize
        '''
        super().__init__(*args,**kwargs)

    def configure(self,hu, input_dim, class_num, 
        dropout,
        dropout_rate,
        output_bias = None, 
        loss="categorical_crossentropy", 
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
            self.params["dropout_rate"] = dropout_rate
        self.params["learning_rate"] = learning_rate

        if not regularize:
            reg = None

        self.model = Sequential()
        self.model.add(Dense(hu,input_dim = input_dim, activity_regularizer=reg, activation="relu",name="Dense1"))
        if dropout:
            self.model.add(Dropout(rate = dropout_rate,seed =self.seeds,name="Dropout1"))
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

    def run(self, dataset, shuffle_when_train = False, gpu = False, set_class_weight = False,
        save_check_point = True, early_stopping= True,
        patience = 5, epochs = 200, batch_size = 10, file = None):
        '''  
        Fit model, store histories, evalute with test data, print report
        
        Params:
        - gpu: True if run with GPU, False if CPU
        - patience: number of epoch to try if no imporvement

        Returns: evalute result on test data
        '''
        x_train = dataset[0]
        y_train = dataset[1]
        x_val = dataset[2]
        y_val = dataset[3]
        x_test = dataset[4]
        y_test = dataset[5]

        self.params["gpu"] = gpu
        self.params["set_class_weight"] = set_class_weight
        self.params["save_check_point"] = save_check_point
        self.params["early_stopping"] = early_stopping
        if early_stopping:
            self.params["patience"] = patience
        self.params["epochs"] = epochs
        self.params["shuffle_when_train"] = shuffle_when_train
        self.params["batch_size"] = batch_size 

        print_params(self.params,"CLASSIFIER PARAMS:",file = file)

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
            class_weight=calculate_weight(
                y_train=np.argmax(y_train,axis=-1)
            )
        
        else:
            class_weight=None

        with tf.device(processor):  # type: ignore 
            self.history = self.model.fit(
                x = x_train,
                y = y_train,
                epochs=epochs,
                verbose="auto", 
                batch_size=batch_size,
                validation_data=(x_val,y_val), 
                shuffle=shuffle_when_train,
                class_weight=class_weight,
                callbacks=callback_list
            )

        self.saved_history = dict(self.history.history)  #type: ignore
    
        with tf.device(processor): # type: ignore 
            self.pred_prob = self.model.predict(x=x_test)
            self.pred_class = np.argmax(self.pred_prob,axis=-1)

            test_results = self.model.evaluate(
                x_test,
                y_test,
                batch_size = batch_size,
                return_dict = True
            )
        y_pred_int = np.argmax(self.pred_prob, axis=1)
        y_true_int = np.argmax(y_test,axis = 1)    #type: ignore

        trp.print_classification_report(y_true=y_true_int,y_pred=y_pred_int, file = file)
        trp.print_confusion_matrix(y_true=y_true_int,y_pred=y_pred_int,file = file)

        return test_results

    
def print_params(params:dict,title:str,file=None):
        '''
        Print parameters 
        '''
        printb("\n=============",file = file)
        printb(title,file = file)
        for key in params.keys():
            printb("{}:{}".format(key,params[key]),file = file)


def split_to_k_folds(fm:FeatureManager, target_col:str, fold_number:int, train_size:float=0.7,
        val_size:float=0.15, categorical_label:bool = True, rebalance = None, file = None):
    '''Split the number to multiple equal size piece, each piece devided to train, validation, test portion
    '''

    fm.params["fold_number"] = fold_number
    fm.params["train_size"] = train_size
    fm.params["val_size"] = val_size
    fm.params["categorical_label"] = categorical_label
    fm.params["rebalance"] = rebalance

    print_params(fm.params,title = "DATA PREPARATION PARAMS:", file=file)

    printb("==========", file= file)
    printb("DATA:", file = file)
    printb("Total rows: {}".format(len(fm.df)),file = file)
    print_labels_distribution(fm.df,target_col=target_col,file=file)

    #Calculate length
    print("Splitting the data...")
    fold_len = int(len(fm.df)/fold_number)
    dataset = []

    for i in range(0,fold_number):
        start_train = i * fold_len
        end_train = start_train + int(fold_len * train_size)
        end_val = end_train + int(fold_len * val_size)
        end_test = end_val + fold_len - int(fold_len * train_size) - int(fold_len * val_size)
        data_train = fm.df.iloc[start_train:end_train]
        data_val = fm.df.iloc[end_train:end_val]
        data_test = fm.df.iloc[end_val:end_test]
        
        match rebalance:
            case "over":
                print("Rebalancing data with over-sampling")
                data_train = tr_utils.over_sampling_rebalance(data=data_train,target_col=target_col)
                data_val = tr_utils.over_sampling_rebalance(data=data_val,target_col=target_col)
            case "under":
                print("Rebalancing data with under-sampling")
                data_train = tr_utils.under_sampling_rebalance(data=data_train,target_col=target_col)
                data_val = tr_utils.under_sampling_rebalance(data=data_val,target_col=target_col)
            case None:  
                pass
            case other:
                print("Failed to rebalance data due to wrong arguments")

        x_train = data_train[fm.cols].values.copy()
        x_val = data_val[fm.cols].values.copy()
        x_test = data_test[fm.cols].values.copy()
    

        if categorical_label:
            y_train = np_utils.to_categorical(data_train[target_col].values,dtype="int64")
            y_val = np_utils.to_categorical(data_val[target_col].values,dtype="int64")
            y_test = np_utils.to_categorical(data_test[target_col].values,dtype="int64")
        else:
            y_train = data_train[target_col].values.copy()
            y_val = data_val[target_col].values.copy()
            y_test = data_test[target_col].values.copy()
        
        printb("\nFOLD {}".format(i+1),file=file)
        printb("Train: {}, Validation: {}, Test: {}".
            format(len(data_train),len(data_val),len(data_test)),file = file)

        printb("\nTrain:", file = file)
        trp.print_labels_distribution(data_train,target_col,file=file)

        printb("\nValidation:", file = file)
        trp.print_labels_distribution(data_val,target_col,file=file)

        printb("\nTest:",file = file)
        trp.print_labels_distribution(data_test,target_col,file=file)

        dataset.append([x_train,y_train,x_val,y_val,x_test,y_test])

    return dataset

def print_features_list(fm: FeatureManager, file = None):
    print("\n=============",file = file)
    print("FEATURES (show 1 for each):",file = file)

    for i in range(0,len(fm.cols),fm.params["lags"]):
        print("{},".format(fm.cols[i]),end=" ",file = file)
    print("\n",file=file)


def evaluate_classifier_k_folds(hu:int,fm: FeatureManager,fold_number:int, gpu:bool=False, 
    save_check_point:bool = True, early_stopping:bool = True, shuffle_when_train:bool = False, set_class_weight:bool = False,
    epochs:int = 200, dropout = True, dropout_rate = 0.3, patience:int = 5,
    batch_size:int = 24, metrics:list[str] = [], write_to_file:bool = False):
    
    #Create a file
    report_file = None
    if write_to_file:
        filename = "../logs/report/{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        report_file = open(filename,'w')

    print_features_list(fm=fm,file=report_file)

    results_list = []

    dataset_list = split_to_k_folds(
        fm = fm,
        categorical_label=True,
        fold_number=fold_number,
        target_col=fm.target_col,
        file = report_file
    )

    for i in range (0,fold_number):
        printb("\n>>>>>> FOLD {}\n".format(i+1), file=report_file)
        callbacks.backend.clear_session()
        classifier = MultiDNNClassifer()        
        
        initial_bias = init_imbalanced_bias(
            y_train = np.argmax(dataset_list[i][1],axis = -1)
        )

        classifier.configure(
            hu = hu, 
            dropout=dropout,
            dropout_rate = dropout_rate,
            input_dim=len(fm.cols),
            output_bias=initial_bias,
            class_num=3,
        )

        processor = "/cpu:0"
        with tf.device(processor):                          # type: ignore 
            test_result = classifier.run(
                gpu = gpu,
                dataset = dataset_list[i],
                epochs=epochs,
                shuffle_when_train = shuffle_when_train,
                patience=patience,
                early_stopping = early_stopping,
                save_check_point = save_check_point,
                set_class_weight = set_class_weight,
                batch_size = batch_size,
                file = report_file)
            results_list.append(test_result)

    trp.print_test_summary(results=results_list,metrics = metrics,file=report_file) #type:_ignore
    
    if report_file is not None:
        report_file.close()

    return results_list

def evaluate_classifier(hu:int,fm: FeatureManager,laps:int, shuffle_before_split= True, gpu:bool=False, 
    save_check_point:bool = True, early_stopping:bool = True, set_class_weight:bool = False,
    epochs:int = 200, dropout = True, dropout_rate = 0.3, shuffle_when_train = False, patience:int = 5,
    batch_size:int = 24, metrics:list[str] = [], write_to_file:bool = False):
    ''' 
    Run the classifier multiple time with loop calculate the average and std value of accuracy and loss

    Params:
    - fm: Feature Manager instance that store features and target values
    - target_col: name of target column
    - laps: number of lap to loop
    
    Return: evaluation metrics
    '''

    #Create a file
    report_file = None
    if write_to_file:
        filename = "../logs/report/{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        report_file = open(filename,'w')

    print_features_list(fm=fm,file=report_file)

    results_list = []

    for i in range (0,laps):
        printb("\n>>>>>> LAP {}\n".format(i+1), file=report_file)
        callbacks.backend.clear_session()
        classifier = MultiDNNClassifer()
        
        dataset = classifier.prepare_data(
            data = fm.df,
            cols = fm.cols,
            shuffle_before_split = shuffle_before_split,
            categorical_label=True,
            random_state=i+1,
            target_col=fm.target_col,
            file = report_file
        )

        initial_bias = init_imbalanced_bias(
            y_train = dataset[1]
        )

        classifier.configure(
            hu = hu, 
            dropout=dropout,
            dropout_rate = dropout_rate,
            input_dim=len(fm.cols),
            output_bias=initial_bias,
            class_num=3,
        )

        processor = "/cpu:0"
        with tf.device(processor):                          # type: ignore 
            test_result = classifier.run(
                gpu = gpu,
                epochs=epochs, 
                patience=patience,
                dataset = dataset,
                shuffle_when_train = shuffle_when_train,
                early_stopping = early_stopping,
                save_check_point = save_check_point,
                set_class_weight = set_class_weight,
                batch_size = batch_size,
                file = report_file)

            results_list.append(test_result)

    trp.print_test_summary(results=results_list,metrics = metrics,file=report_file) #type:_ignore
    
    if report_file is not None:
        report_file.close()

    return results_list
