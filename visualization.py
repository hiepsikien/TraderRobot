import matplotlib.pyplot as plt
from random import randint

def visualize_efficiency_by_cutoff(data,min_delta,max_delta):

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

def visualize_test_results_by_lap(laps,acc_list,cov_list):
    laps = range(len(acc_list))
    plt.figure()
    plt.plot(laps, acc_list, "b", label="Accuracy")
    plt.plot(laps, cov_list, "r", label="Coverage")
    plt.title("Accuracy and Coverage")
    plt.xlabel("Test Lap")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

def scatter_test_results(acc_list,cov_list):
    plt.figure(figsize=(6,6))
    
    plt.scatter(acc_list, cov_list, s=80, c="b", alpha=0.5)
    
    plt.scatter(sum(acc_list)/len(acc_list),sum(cov_list)/len(cov_list), s=100, c="r")

    for i in range(0,len(acc_list)):
        ran_x = randint(-10,10) * 0.0001
        ran_y = randint(-10,10) * 0.0001
        plt.annotate(i+1,(acc_list[i]+ran_x,cov_list[i]+ran_y))
        
    plt.title("Accurracy and Coverage")
    plt.xlabel("Accuracy")
    plt.ylabel("Coverage")
    plt.show()