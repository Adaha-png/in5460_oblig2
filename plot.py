import matplotlib.pyplot as plt
import numpy as np

#----- Reader of Acc and loss Lists
#Changed manually after models have been created
with open("lstmLosslist.txt") as f:
    file = f.read()
file = file.split("; ")
file.pop()
for i in range(len(file)):
    file[i] = float(file[i])
timestep = np.arange(len(file))

#---- Ground Truth reader ------
# with open("lstmGroundTruth") as g:
#     file2 = g.read()
# file2 = file2.split("; ")
# file2.pop()
# for i in range(len(file2)):
#     file2[i] = float(file2[i])
# timestep2 = np.arange(len(file2))

#-------------------------------------------------------------------------
#Ploting LSTM accuracy during classification training
# plt.plot(timestep, file, "r", label="Accuracy")
# plt.grid()
# plt.title("Classification model using LSTM")
# plt.xlabel("Training timesteps", fontdict={"size":12, "family":"serif"})
# plt.ylabel("Accuracy", fontdict={"size":12, "family":"serif"})
# plt.legend()
# plt.show()

#-------------------------------------------------------------------------
#Plotting RNN accuracy during classification training
#-------------------------------------------------------------------------
# plt.plot(timestep, file, "r", label="Accuracy")
# plt.grid()
# plt.title("Classification model using RNN")
# plt.xlabel("Timestep", fontdict={"size":12, "family":"serif"})
# plt.ylabel("Accuracy", fontdict={"size":12, "family":"serif"})
# plt.legend()
# plt.show()


#-------------------------------------------------------------------------
#Plotting LSTM loss during prediction training
#-------------------------------------------------------------------------
plt.plot(timestep, file, "r", label="Mean Squared Error")
plt.grid()
plt.title("Prediction model using LSTM")
plt.xlabel("Timestep", fontdict={"size":12, "family":"serif"})
plt.ylabel("Loss", fontdict={"size":12, "family":"serif"})
plt.legend()
plt.show()


#-------------------------------------------------------------------------
#Plotting RNN loss during prediction training
#-------------------------------------------------------------------------
# plt.plot(timestep, file, "r", label="Mean Squared Error")
# plt.grid()
# plt.title("Prediction model using RNN")
# plt.xlabel("Timestep", fontdict={"size":12, "family":"serif"})
# plt.ylabel("Loss", fontdict={"size":12, "family":"serif"})
# plt.legend()
# plt.show()


#-------------------------------------------------------------------------
#Plotting LSTM vs Ground Truth
#-------------------------------------------------------------------------
# plt.plot(timestep, file, 'g', label = "Ground truth")
# plt.plot(timestep2, file2, 'b', label = "Predicted output")
# plt.grid()
# plt.title("Predicted vs Ground Truth using LSTM")
# plt.xlabel("Timestep", fontdict={"size":12, "family":"serif"})
# plt.ylabel("kWh", fontdict={"size":12, "family":"serif"})
# plt.legend()
# plt.show()


#-------------------------------------------------------------------------
#Plotting RNN vs Ground Truth
#-------------------------------------------------------------------------
# plt.plot(timestep, file, 'g', label = "Ground truth")
# plt.plot(timestep2, file2, 'b', label = "Predicted output")
# plt.grid()
# plt.title("Predicted vs Ground Truth using RNN")
# plt.xlabel("Timestep", fontdict={"size":12, "family":"serif"})
# plt.ylabel("kWh", fontdict={"size":12, "family":"serif"})
# plt.legend()
# plt.show()



            

            

            