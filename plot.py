import matplotlib.pyplot as plt


def plotter (timestep, loss,classification, rnn):
    if(classification):
        if(rnn):
            plt.plot(timestep, loss, "r", label="CrossEntropy loss")
            plt.grid()
            plt.title("Classification model using RNN")
            plt.xlabel("Timestep", fontdict={"size":12, "family":"serif"})
            plt.ylabel("Loss", fontdict={"size":12, "family":"serif"})
            plt.legend()
            plt.show()
        else:
            plt.plot(timestep, loss, "r", label="CrossEntropy loss")
            plt.grid()
            plt.title("Classification model using LSTM")
            plt.xlabel("Timestep", fontdict={"size":12, "family":"serif"})
            plt.ylabel("Loss", fontdict={"size":12, "family":"serif"})
            plt.legend()
            plt.show()
    else:
        if(rnn):
            plt.plot(timestep, loss, "r", label="Mean Squared Error")
            plt.grid()
            plt.title("Prediction model using RNN")
            plt.xlabel("Timestep", fontdict={"size":12, "family":"serif"})
            plt.ylabel("Loss", fontdict={"size":12, "family":"serif"})
            plt.legend()
            plt.show()
        else:
            plt.plot(timestep, loss, "r", label="Mean Squared Error")
            plt.grid()
            plt.title("Prediction model using LSTM")
            plt.xlabel("Timestep", fontdict={"size":12, "family":"serif"})
            plt.ylabel("Loss", fontdict={"size":12, "family":"serif"})
            plt.legend()
            plt.show()
