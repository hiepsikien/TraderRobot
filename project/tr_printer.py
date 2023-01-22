import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

DEFAULT_PRINT_METRICS = [
    "loss",
    "accuracy",
    "precision",
    "recall",
    "precision-0.90",
    "recall-0.90"
]


def printb(var,file, end = None):
    '''Print to both stdout and file'''
    print(var,end=end)
    print(var,end=end,file=file)
    

def print_test_summary(results:list[dict],metrics:list[str],file = None):
    '''
    Print classifier evaluation summary

    Params:
    - results: list of metrics scores returns by classifier for tested laps
    - cols: list of metrics to show
    '''
    printb("\n>>>>>>",file = file)
    printb("EVALUATION SUMMARY:",file = file)
    
    data = pd.DataFrame(data=results,index=[i for i in range(1,len(results)+1)])

    df = data.copy()

    df.loc["mean"] = data.mean()
    df.loc["std"] = data.std()
    df.loc["min"] = data.min()
    df.loc["max"] = data.max()

    df = df.astype(float).round(3)
    
    if metrics is not []:
        printb(df[metrics],file=file)
    else:
        printb(df["loss","accuracy","precision","recall"],file=file)

def print_labels_distribution(data:pd.DataFrame,target_col:str, file=None):
    value_counts = data[target_col].value_counts()
    value_counts.sort_index(inplace=True)
    for i in value_counts.index:
        printb("Label {}: {}({}%)".format(
            i,
            value_counts[i],                                    #type: ignore
            round(value_counts[i]/value_counts.sum()*100,2)     #type: ignore
            ),
            file = file
        )
def print_classification_report(y_true,y_pred,file=None):
    '''
    Print classification report
    '''
    printb("\n=============",file = file)
    printb("CLASSIFICATION REPORT:",file = file)
    printb(classification_report(y_true, y_pred),file = file)

def print_confusion_matrix(y_true,y_pred, file = None):
    '''
    Print confusion matrix
    '''
    printb("\n=============",file = file)
    printb("CONFUSION MATRIX:",file = file)
    con_matrix = confusion_matrix(y_true=y_true,y_pred=y_pred) #type: ignore
    df = pd.DataFrame(con_matrix, columns = ["P{}".format(i) for i in range(0, len(con_matrix))])
    df.loc[:,"Total"]= df.sum(axis = 1, numeric_only = True).astype(int)
    df.loc["Total"] = df.sum(axis = 0, numeric_only = True).astype(int)
    for i in range(0, len(con_matrix)):
        df["RP{}".format(i)] = round(df["P-{}".format(i)]/df["Total"],3)
    printb(df,file = file)