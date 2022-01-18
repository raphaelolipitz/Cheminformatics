#install all packages for the predictor

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
import pandas as pd
from IPython.core.display import HTML
import os
from rdkit import RDConfig
from sklearn.model_selection import train_test_split
from numpy import loadtxt
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.jit import script, trace       # hybrid frontend decorator and tracing jit



SDFFile = "/Users/raphael/Documents/Studium/Vorlesungen/Cheminfo/Cheminformaticproject/Cheminformatics/qspr-dataset-02.sdf 2"
Data = PandasTools.LoadSDF(SDFFile,smilesName='SMILES',
                           molColName='Molecule',
                           includeFingerprints=True,
                           removeHs=False, strictParsing=True)


def datapreprossessing():
    #compute the SMILES repressentation for the molecules
    smi = []
    mols = [mol for mol in Chem.SDMolSupplier(SDFFile)]
    for mol in mols:
        smi.append(Chem.MolToSmiles(mol))

    #calculate the MorgenFingerprint as Vector for the molocules? Maybe a nother Fingerprint would be better.
    fp = []
    for mol in mols:
        fp.append(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits = 124))

    Data["SMILES"] =smi
    Data["Fingerprints"] = fp

    print(Data.info())
    print(Data.head())



    #first I want to split the data into a train a validation and a test set with the sklearn packege.
    #one of the common ratios for splitting the data.
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    #because the package can only split between test and train I used the function twice.
    x_train, x_test, y_train, y_test = train_test_split(Data, Data, test_size= 1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size= test_ratio / (test_ratio + validation_ratio))

    print(x_train, x_val, x_test)

    return Data

# silmple model for pytorch to optimize the network for our purpose.
class Neural_Network(nn.Module):

    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # 2 X 3 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)  # 3 X 1 tensor

    def forward(self, X):
        self.z = torch.matmul(X, self.W1)  # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted))
        print("Output: \n" + str(self.forward(xPredicted)))






NN = Neural_Network()
for i in range(1000):  # trains the NN 1,000 times
    print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
NN.saveWeights(NN)
NN.predict()


#start of the funktions
def main():
    datapreprossessing()



#start of the program
if __name__ == "__main__":
    main()
