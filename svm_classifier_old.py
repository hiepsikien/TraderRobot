import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import product

class SVMClassifier():

    def __init__(self, scaler = 0, test_ratio = 0.3, test_laps =5) -> None:
        self.prepared_data = None
        self.test_ratio =test_ratio
        self.test_laps = test_laps
        self.scaler = scaler
        self.svc = svm.SVC()

    def set_scaler(self,scaler):
        self.scaler = scaler

    def set_test_lap(self,test_laps):
        self.test_laps = test_laps

    def set_test_ratio(self,test_ratio):
        self.test_ratio = test_ratio
    
    def prepare_data(self,symbol,sub_symbol, interval, lags, sub_lags, vol_lags):
 
        if self.scaler == 0:
            my_scaler = None
        elif self.scaler == 1:
            my_scaler = MaxAbsScaler()
        elif self.scaler == 2:
            my_scaler = MinMaxScaler()
        elif self.scaler == 3:
            my_scaler = StandardScaler()
        else:
            my_scaler = None

        #Import target currency
        data = pd.read_csv("{}-{}.csv".format(symbol,interval), index_col="Open Time")
        data = data[["Close","Volumn"]]
        data["Log_Return"] = np.log(data["Close"].div(data["Close"].shift(1)))
        data["Direction"] = np.sign(data["Log_Return"])


        cols = []
        
        for lag in range(1,lags+1):
            col = "Lag_{}".format(lag)
            data[col] = data["Log_Return"].shift(lag)
            cols.append(col)

        #Add volumn info
        if vol_lags > 0:
            data["Log_Volumn_Change"] = np.log(data["Volumn"].div(data["Volumn"].shift(1)))
            for lag in range(1,vol_lags+1):
                col = "Volumn_Lag{}".format(lag)
                data[col] = data["Log_Volumn_Change"].shift(lag)
                cols.append(col)

        #Import correlated currency

        if sub_symbol != None:

            sub_data = pd.read_csv("{}-{}.csv".format(sub_symbol,interval), index_col="Open Time")
            sub_data = sub_data[["Close"]]
            sub_data.columns = ["Sub_Close"]
            sub_data["Log_Sub_Return"] = np.log(sub_data["Sub_Close"].div(sub_data["Sub_Close"].shift(1)))
            
            for lag in range(1,sub_lags+1):
                col = "Sub_Lag_{}".format(lag)
                data[col] = sub_data["Log_Sub_Return"].shift(lag)
                cols.append(col)

            #Merge target and correlated currencies info to a big dataframe
            self.prepared_data = pd.concat([data, sub_data], join="inner", axis = 1)
            self.prepared_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            # self.prepared_data.to_csv("merged.csv")
        else:
            self.prepared_data = data

        self.prepared_data.dropna(inplace=True)

        #Nomarlize data
        if (my_scaler!=None):
            normalized_cols = []
            for col in cols:
                # print("Nomalizing col: {}".format(col))
                normalized_col = "Nom_" + col
                array = self.prepared_data[col].to_frame()
                try:
                    my_scaler.fit(array)
                    self.prepared_data[normalized_col] = my_scaler.transform(array)
                    normalized_cols.append(normalized_col)
                except:
                    print("An exception occured")
            return self.prepared_data, normalized_cols
        else:
            return self.prepared_data, cols

    def run(self,data,cols):
        accuracy_scores =[]
        for i in range(0,self.test_laps):
            print("Lap {}: ".format(i+1))
            x_train, x_test, y_train, y_test = train_test_split(data[cols],data["Direction"],test_size=self.test_ratio)
            self.svc.fit(X = x_train, y = y_train)
            print("Trained. Testing...")
            y_pred = self.svc.predict(X = x_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            print("Accuracy Score: {}".format(accuracy))
        score_array = np.array(accuracy_scores)
        mean = score_array.mean()
        std = score_array.std()
        print("Average: {}, Std: {}".format(mean,std))
        return mean, std

    def prepare_and_run(self,symbol,sub_symbol=None,interval="1h",lags=8,sub_lags=0,vol_lags=0):
        print("Symbol={}, Sub_Symbol={}, Interval={}, Lags={}, Sub_Lags={}, Vol_Lags={}".format(symbol, sub_symbol, interval,lags,sub_lags,vol_lags))
        (data, cols) = self.prepare_data(symbol=symbol, sub_symbol=sub_symbol,interval=interval,lags=lags,sub_lags=sub_lags,vol_lags=vol_lags)
        (mean, std) = self.run(data=data,cols=cols)
        return(mean,std)

    def optimize(self,symbol,sub_symbol,interval,lags,sub_lags,vol_lags,scaler):
        lag_range = range(1,lags+1)
        sub_lag_range = range(0,sub_lags+1)
        vol_lag_range = range(0,vol_lags+1)
        scaler_range = range(0,scaler+1)

        combs = list(product(lag_range,sub_lag_range,vol_lag_range,scaler_range))

        results = []
        
        for comb in combs:
            (lags, sub_lags, vol_lags, scaler) = comb
            (mean, std) = self.prepare_and_run(symbol=symbol,sub_symbol=sub_symbol,interval=interval,lags=lags, sub_lags=sub_lags, vol_lags=vol_lags)
            result = (lags,sub_lags,vol_lags,scaler,mean,std)
            results.append(result)
            
        return results