#install all packages for the predictor
import torch
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





    #first I want to split the data into a train a validation and a test set with the sklearn packege.
    #one of the common ratios for splitting the data.
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    #because the package can only split between test and train I used the function twice.
    x_train, x_test, y_train, y_test = train_test_split(Data, Data, test_size= 1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size= test_ratio / (test_ratio + validation_ratio))



    return Data

# silmple model for pytorch to optimize the network for our purpose.
class Neural_Network(nn.Module):

    def __init__(self, ):
        super(Neural_Network, self).__init__()

        # parameters
        self.inputSize = 124
        self.outputSize = 1
        self.hiddenSize = 124

        # Defined Layer imputlayer is as big as the Fingerprint(124bits). Outputlayer is 1 for one LD50 value.
        self.input_layer = nn.Linear(self.inputSize, self.hiddenSize)
        self.hidden_layer = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.output_layer = nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, x):
        #for the acivation ReLu seamse to be a good chois
        x = F.relu(self.input_layer)
        x = F.relu(self.hidden_layer)
        x = self.output_layer
        return x










#start of the funktions
def main():
    datapreprossessing()


    Predictor = Neural_Network()
    print(Predictor)
    optimizer = optim.Adam(Predictor.parameters(), lr = 0.001)

    epochs = 1000

    for i in range(epochs):  # trains the NN 1,000 times
        X , y = data
        Predictor.zero_grad()




#start of the program
if __name__ == "__main__":
    main()
