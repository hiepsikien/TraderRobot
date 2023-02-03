import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from feature_manager import FeatureManager
from tr_printer import printb, print_params, print_labels_distribution, print_test_summary, print_classification_report, print_confusion_matrix
from tr_utils import over_sampling_rebalance, under_sampling_rebalance, init_imbalanced_bias
from keras.utils import np_utils
import datetime
from classifier.multi_dnn_classifier import MultiDNNClassifer
from feature_manager import FeatureManager
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report


DEFAULT_SVM_METRICS = [
    'accuracy', 
    '0.0_precision',
    '0.0_recall',
    '0.0_f1-score',
    '1.0_precision',
    '1.0_recall',
    '1.0_f1-score',
    '2.0_precision',
    '2.0_recall',
    '2.0_f1-score',
]

class CrossValidator():
    '''
    A class to run cross validation for classifier including MultiDNN and MultiSVM
    '''
    def __init__(self) -> None:
        self.params= dict()

    def split_time_series(self,fm:FeatureManager, target_col:str, fold_number:int, 
        categorical_label:bool = True, rebalance = None, file = None):
        '''
        Split to k_fold as timeseries
        Example: if folds is 3, the data will be devided into 5 equal parts. 
        Fold 1: Train 1, Val 2, Test 3
        Fold 2: Train 1+2, Val 3, Test 4
        Fold 3: Train 1+2+3, Val 4, Test 5

        Params:
        - fold_number: number of data portion to be splitted
        ..

        Returns: list of datasets 
        '''

        printb("\n============", file = file)
        printb("DATA:", file = file)
        printb("Total rows: {}".format(len(fm.df)),file = file)
        print_labels_distribution(fm.df[target_col],file = file)

        #Calculate length
        print("Splitting the data...")
        fold_len = int(len(fm.df)/(fold_number+2))
        dataset_list = []

        for i in range(0,fold_number):
            data_train = fm.df.iloc[0:(i+1)*fold_len]
            data_val = fm.df.iloc[(i+1)*fold_len:(i+2)*fold_len]
            data_test = fm.df.iloc[(i+2)*fold_len:(i+3)*fold_len]
            
            match rebalance:
                case "over":
                    print("Rebalancing data with over-sampling")
                    data_train = over_sampling_rebalance(data = data_train,target_col = target_col)
                    # data_val = tr_utils.over_sampling_rebalance(data=data_val,target_col=target_col)
                case "under":
                    print("Rebalancing data with under-sampling")
                    data_train = under_sampling_rebalance(data = data_train,target_col = target_col)
                    # data_val = tr_utils.under_sampling_rebalance(data=data_val,target_col=target_col)
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

            dataset_list.append([x_train,y_train,x_val,y_val,x_test,y_test])

        return dataset_list

    def split_time_series_no_test(self,fm:FeatureManager, target_col:str, fold_number:int, 
        categorical_label:bool = True, rebalance = None, file = None):
        '''
        Split to k_fold as timeseries
        Example: if folds is 3, the data will be devided into 4 equal parts. 
        Fold 1: Train 1, Val 2
        Fold 2: Train 1+2, Val 3
        Fold 3: Train 1+2+3, Val 4

        Params:
        - fold_number: number of data portion to be splitted
        ..

        Returns: list of datasets 
        '''
        
        printb("\n============", file = file)
        printb("DATA:", file = file)
        printb("Total rows: {}".format(len(fm.df)),file = file)
        print_labels_distribution(fm.df[target_col],file = file)

        #Calculate length
        print("Splitting the data...")
        fold_len = int(len(fm.df)/(fold_number+1))
        dataset_list = []

        for i in range(0,fold_number):
            data_train = fm.df.iloc[0:(i+1)*fold_len]
            data_val = fm.df.iloc[(i+1)*fold_len:(i+2)*fold_len]
            
            match rebalance:
                case "over":
                    print("Rebalancing data with over-sampling")
                    data_train = over_sampling_rebalance(data = data_train,target_col = target_col)
                    # data_val = tr_utils.over_sampling_rebalance(data=data_val,target_col=target_col)
                case "under":
                    print("Rebalancing data with under-sampling")
                    data_train = under_sampling_rebalance(data = data_train,target_col = target_col)
                    # data_val = tr_utils.under_sampling_rebalance(data=data_val,target_col=target_col)
                case None:  
                    pass
                case other:
                    print("Failed to rebalance data due to wrong arguments")

            x_train = data_train[fm.cols].values.copy()
            x_val = data_val[fm.cols].values.copy()

            if categorical_label:
                y_train = np_utils.to_categorical(data_train[target_col].values,dtype="int64")
                y_val = np_utils.to_categorical(data_val[target_col].values,dtype="int64")
            else:
                y_train = data_train[target_col].values.copy()
                y_val = data_val[target_col].values.copy()

            dataset_list.append([x_train,y_train,x_val,y_val])

        return dataset_list

    def split_equal(self,fm:FeatureManager, target_col:str, fold_number:int, train_size:float=0.7,
            val_size:float = 0.15, categorical_label:bool = True, rebalance = None, file = None):
        '''
        Split the number to multiple equal size piece, each piece devided to train, validation, test portion

        Params:
        - fold_number: number of data portion to be splitted
        ..

        Returns: list of datasets 
        '''

        fm.params["fold_number"] = fold_number
        fm.params["train_size"] = train_size
        fm.params["val_size"] = val_size
        fm.params["categorical_label"] = categorical_label
        fm.params["rebalance"] = rebalance

        print_params(fm.params,title = "DATA PREPARATION PARAMS:",file=file)

        printb("\n============", file = file)
        printb("DATA:", file = file)
        printb("Total rows: {}".format(len(fm.df)),file = file)
        print_labels_distribution(fm.df[target_col],file = file)

        #Calculate length
        print("Splitting the data...")
        fold_len = int(len(fm.df)/fold_number)
        dataset_list = []

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
                    data_train = over_sampling_rebalance(data = data_train,target_col = target_col)
                    # data_val = tr_utils.over_sampling_rebalance(data=data_val,target_col=target_col)
                case "under":
                    print("Rebalancing data with under-sampling")
                    data_train = under_sampling_rebalance(data = data_train,target_col = target_col)
                    # data_val = tr_utils.under_sampling_rebalance(data=data_val,target_col=target_col)
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

            dataset_list.append([x_train,y_train,x_val,y_val,x_test,y_test])

        return dataset_list

    def print_features_list(self,fm: FeatureManager, file = None):
        print("\n=============",file = file)
        print("FEATURES (show 1 for each):",file = file)

        for i in range(0,len(fm.cols),fm.params["lags"]):
            print("{},".format(fm.cols[i]),end=" ",file = file)
        print("\n",file=file)

    def cross_validate_multi_svm(self,fm:FeatureManager, C:int=4, kernel:str="poly",degree:int=4, gamma="scale",
            random_state:int=1,cache_size:int=3000,probability:bool=True, fold_number:int=10,decision_function_shape:str="ovr",
            class_weight:str="balanced",tol:float=1e-3,rebalance = None,train_size:float=0.7,split_type:str="time_series_split",
            write_to_file:bool = False):
        ''' Cross validate multi category SVM classisifer
        '''

        #Create a file
        report_file = None
        if write_to_file:
            filename = "../logs/report/{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            report_file = open(filename,'w')

        self.print_features_list(
            fm = fm,
            file = report_file
        )

        self.params["classifier"] = "svm"
        self.params["C"] = C
        self.params["kernel"] = kernel
        self.params["degree"] = degree
        self.params["gamma"] = gamma
        self.params["randome_state"] = random_state
        self.params["probability"] = probability
        self.params["fold_numeber"] = fold_number
        self.params["decision_function_shape"] = decision_function_shape
        self.params["class_weight"] = class_weight
        self.params["split_type"] = split_type
        self.params["rebalance"] = rebalance

        print_params(params =self.params,title="CROSS VALIDATION PARAMS",file=report_file)

        # Split the data
        match split_type:
            case "equal_split":
                dataset_list = self.split_equal(
                    fm = fm,
                    categorical_label = False,
                    fold_number = fold_number,
                    target_col = fm.target_col,
                    rebalance = rebalance,
                    file = report_file,
                    train_size=train_size,
                    val_size=0.0
                )
            case "time_series_split":
                dataset_list = self.split_time_series_no_test(
                    fm = fm,
                    categorical_label = False,
                    fold_number = fold_number,
                    target_col = fm.target_col,
                    rebalance = rebalance,
                    file = report_file
                )
            case other:
                raise ValueError("'Split type' must be 'equal_split' or 'time_series_split'")
        
        self.result_list = []

        for i in range(0,fold_number):
            printb("\n>>>>>> FOLD {}\n".format(i+1), file = report_file)

            classifier = svm.SVC(
                C=C,
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                random_state=random_state,
                cache_size=cache_size,
                probability=probability,
                verbose = True,
                decision_function_shape = decision_function_shape,
                class_weight = class_weight,
                tol=tol
            )
            
            [x_train, y_train, x_val, y_val] = dataset_list[i]

            printb("\nDATA IN FOLD",file=report_file)
            printb("Train: {}, Validation: {}".
            format(len(y_train),len(y_val)),file = report_file)

            printb("\nTrain:", file = report_file)
            print_labels_distribution(y_train,file=report_file)

            printb("\nTest:",file = report_file)   
            print_labels_distribution(y_val,file=report_file)

            classifier.fit(X=x_train,y=y_train)
            y_pred = classifier.predict(x_val)

            result_dict = classification_report(y_pred=y_pred,y_true=y_val,output_dict=True)
            df = pd.json_normalize(result_dict, sep='_')             #type: ignore
            result_dict = df.to_dict(orient='records')[0]
            self.result_list.append(result_dict)

            print_classification_report(y_pred=y_pred,y_true=y_val,file=report_file)
            print_confusion_matrix(y_pred=y_pred,y_true=y_val,file=report_file)
        
        print_test_summary(results=self.result_list,metrics = DEFAULT_SVM_METRICS,file=report_file)
            

    def cross_validate_multi_dnn(self,hu:int,fm: FeatureManager,fold_number:int, gpu:bool=False, 
        save_check_point:bool = True, early_stop:bool = True, rebalance = None, split_type:str="time_series_split", set_initial_bias:bool = True, shuffle_when_train:bool = False, set_class_weight:bool = False,
        epochs:int = 200, dropout = True, dropout_rate = 0.3, patience:int = 5,
        batch_size:int = 24, metrics:list[str] = [], write_to_file:bool = False):
        ''' 
        Split the data to k equal parts, each part split to train, validation, test.
        Run the classifier on each parts.

        Params:
        - fold_number: number of fold
        - shuffle_when_train: to shuffle the data when training or not

        Returns: result list
        '''

        fm.params["split_type"] = split_type

        #Create a file
        report_file = None
        if write_to_file:
            filename = "../logs/report/{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            report_file = open(filename,'w')

        self.print_features_list(
            fm = fm,
            file = report_file
        )

        results_list = []

        match split_type:
            case "equal_split":
                dataset_list = self.split_equal(
                    fm = fm,
                    categorical_label = True,
                    fold_number = fold_number,
                    target_col = fm.target_col,
                    rebalance = rebalance,
                    file = report_file
                )
            case "time_series_split":
                dataset_list = self.split_time_series(
                    fm = fm,
                    categorical_label = True,
                    fold_number = fold_number,
                    target_col = fm.target_col,
                    rebalance = rebalance,
                    file = report_file
                )
            case other:
                raise ValueError("'Split type' must be 'equal_split' or 'time_series_split'")

        for i in range (0,fold_number):
            printb("\n>>>>>> FOLD {}\n".format(i+1), file = report_file)
            classifier = MultiDNNClassifer()
            
            if set_initial_bias:
                initial_bias = init_imbalanced_bias(
                    y_train = np.argmax(dataset_list[i][1],axis = -1)
                )
            else:
                initial_bias = None

            classifier.configure(
                hu = hu, 
                dropout=dropout,
                dropout_rate = dropout_rate,
                input_dim = len(fm.cols),
                output_bias = initial_bias,
                class_num = 3,
            )

            processor = "/cpu:0"
            with tf.device(processor):                          # type: ignore 
                test_result = classifier.run(
                    gpu = gpu,
                    dataset = dataset_list[i],
                    epochs = epochs,
                    shuffle_when_train = shuffle_when_train,
                    patience=patience,
                    early_stop = early_stop,
                    save_check_point = save_check_point,
                    set_class_weight = set_class_weight,
                    batch_size = batch_size,
                    file = report_file)
                results_list.append(test_result)

        print_test_summary(
            results = results_list,
            metrics = metrics,
            file = report_file
        ) 
        
        if report_file is not None:
            report_file.close()

        return results_list

