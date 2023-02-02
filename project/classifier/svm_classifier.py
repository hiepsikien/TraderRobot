import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

class SVMClassifier():

    def __init__(self, test_ratio = 0.3, neg_cutoff=0.47, pos_cutoff=0.53, test_laps = 5) -> None:
        self.test_ratio =test_ratio
        self.svc = svm.SVC()
        self.test_laps = test_laps
        self.neg_cutoff = neg_cutoff
        self.pos_cutoff = pos_cutoff
        self.svc.probability = True
        self.df = None

    def set_test_laps(self,test_laps):
        self.test_laps = test_laps

    def set_test_ratio(self,test_ratio):
        self.test_ratio = test_ratio

    def run(self,data,cols, target_col):
        accuracy_scores =[]
        coverage_scores =[]

        predict_probs = []

        x_tests = []

        for i in range(0,self.test_laps):
            print("=======")
            print("Lap {}: ".format(i+1))
            x_train, x_test, y_train, y_test = train_test_split(data[cols],data[target_col],test_size=self.test_ratio)
            self.svc.fit(X = x_train, y = y_train)
            print("Trained. Testing...")

            pred_prob = self.svc.predict_proba(X = x_test)

            df = pd.DataFrame(data=pred_prob, columns=["neg","pos"])
            df["dir"] = np.array(y_test)
            
            all_num = df.size

            pred_num = df.loc[(df["pos"]>self.pos_cutoff) | (df["pos"]<self.neg_cutoff)].size
            correct_num = df.loc[(
                ((df["pos"] > self.pos_cutoff) & (df["dir"] == 1)) |
                 ((df["pos"] < self.neg_cutoff) & (df["dir"] == 0))
                 )].size

            acc = correct_num / pred_num
            cov = pred_num / all_num

            accuracy_scores.append(acc)
            coverage_scores.append(cov)

            predict_probs.append(pred_prob)
            x_tests.append(x_test)
            
            print("Correct = {}, Predict = {}, All = {} ".format(correct_num,pred_num,all_num))
            print("Accuracy Score: {}, Coverage Score: {}".format(round(acc,3), round(cov,3)))

            self.df = df

        acc_arr = np.array(accuracy_scores)
        cov_arr = np.array(coverage_scores)
        
        print("Accuracy Score Average: {}, Std: {}".format(acc_arr.mean(),acc_arr.std()))
        print("Coverage Score Average: {}, Std: {}".format(cov_arr.mean(),cov_arr.std()))