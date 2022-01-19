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
    Data.info()
    #first I want to split the data into a train a validation and a test set with the sklearn packege.
    #one of the common ratios for splitting the data.
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    #because the package can only split between test and train I used the function twice.
    x_train, x_test, y_train, y_test = train_test_split(Data, Data, test_size= 1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size= test_ratio / (test_ratio + validation_ratio))

    print(x_train, x_test, y_train, y_test)


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


    #loss function from torch for Mean Square Function
    loss = nn.MSELoss()

    # optimizer for the gradient decent.
    optimizer = optim.Adam(Predictor.parameters(), lr=0.001)

    # train the NN-Model
    number_of_epochs = 1000

    for epoch in range(number_of_epochs):  # trains the NN 1,000 times.
        X , y = data

        #Force Optimizer to do 1-Step.
        optimizer.step()

        # Backpropagation for the learningprocess.
        loss.backward()

        #IMPORTEND: After the Optimizer making a step the gradient must set to zero.
        Predictor.zero_grad()
        
        #calculating and printing the Loss and the accuraacy of the Predictor every 10 epochs.
        if epoch % 10 == 0:
            train_acc = calculate_accuracy(y_train, y_pred)
            y_test_pred = net(X_test)
            y_test_pred = torch.squeeze(y_test_pred)
            test_loss = criterion(y_test_pred, y_test)
            test_acc = calculate_accuracy(y_test, y_test_pred)
            print(
                f'''epoch {epoch}
        Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
        Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
        ''')
    #safing our trained Predictor.
    MODEL_PATH = 'predicter-model.pth'
    torch.save(Predictor, MODEL_PATH)


    # resoring the model for using it.
    #Predictor = torch.load(MODEL_PATH)


#start of the program
if __name__ == "__main__":
    main()
