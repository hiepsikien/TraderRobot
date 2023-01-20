import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import confusion_matrix
import numpy as np

def visualize_efficiency_by_cutoff(data,min_delta,max_delta):
    if data is not None:
        pd = data.loc[((data["delta"] >= min_delta) & (data["delta"] <= max_delta))]
        del_list = pd["delta"].tolist()
        acc_list = pd["accuracy"].tolist()
        cov_list = pd["coverage"].tolist()

        fig, ax = plt.subplots()

        ax.plot(del_list, acc_list, "b", label="Accuracy")
        ax.set_xlabel("Delta")
        ax.set_ylabel("Accuracy")
        
        ax2 = ax.twinx()
        ax2.plot(del_list, cov_list, "r", label="Coverage")
        ax2.set_ylabel("Coverage")

        plt.title("Accuracy and Coverage by Cutoff Delta")
        plt.legend()
        plt.show()
    print("Data is none")

def visualize_test_results_by_lap(acc_list,cov_list):
    laps = range(len(acc_list))
    plt.figure()
    plt.plot(laps, acc_list, "b", label="Accuracy")
    plt.plot(laps, cov_list, "r", label="Coverage")
    plt.title("Accuracy and Coverage")
    plt.xlabel("Test Lap")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

def visualize_scatter(x_list,y_list,x_label,y_label,title,add_point_label = False):
    plt.figure(figsize=(6,6))
    plt.scatter(x_list, y_list, s=80, c="b", alpha=0.5)
    plt.scatter(sum(x_list)/len(x_list),sum(y_list)/len(y_list), s=100, c="r")

    if add_point_label:
        for i in range(0,len(x_list)):
            ran_x = randint(-10,10) * 0.0001
            ran_y = randint(-10,10) * 0.0001
            plt.annotate(i+1,(x_list[i]+ran_x,y_list[i]+ran_y))
            
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(7,7))
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(cm, annot=True, fmt="d")
