#install all packages for the predictor
import torch
from pandas.io import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, MolFromSmiles, AddHs
from rdkit.Chem import PandasTools
from rdkit.Chem import rdchem
import pandas as pd
from IPython.core.display import HTML
import os
from rdkit import RDConfig
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import Lipinski
from sklearn.model_selection import train_test_split
from numpy import loadtxt
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.jit import script, trace       # hybrid frontend decorator and tracing jit
'''
import argparse

parser = argparse.ArgumentParser(description='Arguments for the Nussinov algorithm')

parser.add_argument('-i', "--imput", type=argparse.FileType('r'))
parser.add_argument("-o", "--output", action='store', dest='output', help="Directs the output to a name of your choice")

args = parser.parse_args()

seq = args.i.readlines()[1:]

args.i.close()
'''



#helper function

def check_logp(dataset):
    all_smiles = dataset["SMILES"]
    logp_sum=0
    total=0
    logp_score_per_molecule=[]
    for smiles in all_smiles:
        new_mol=Chem.MolFromSmiles(smiles)
        try:
            val = Crippen.MolLogP(new_mol)
        except:
            continue
        logp_sum+=val
        logp_score_per_molecule.append(val)
        total+=1
    return logp_sum/total, logp_score_per_molecule

def logS(logP, MWT,RB,AP):
    return 0.16-0.63*logP-0.0062*MWT+0.066*RB-0.74*AP

#calculating the solubility of the molecules
def check_logS(dataset):

    all_smiles = dataset['SMILES']

    logS_values = []
    total, logP_values = check_logp(dataset)


    c = 0
    for smiles in all_smiles:
        mol = AddHs(MolFromSmiles(smiles))

        MWT = ExactMolWt(mol)
        RB = Chem.Lipinski.NumRotatableBonds(mol)
        AP = len(list(mol.GetAromaticAtoms())) / mol.GetNumAtoms(onlyExplicit=True)
        logP = logP_values[c]
        c+=1
        logS_values.append(logS(logP,MWT,RB,AP))



    return logS_values


#cacking if molecule if it have a aromatic ring.
def ceck_aromaticity(dataset):
    all_smiles = dataset['SMILES']

    aromaticities = []
    for smiles in all_smiles:
        mol = MolFromSmiles(smiles)
        num = mol.GetNumAtoms()
        temp = []
        for i in range(num):
            is_aromatic = mol.GetAtomWithIdx(i).GetIsAromatic()
            if is_aromatic == True:
                temp.append(1)
                break
            if is_aromatic == False:
                temp.append(0)
        if 1 in temp:
            aromaticities.append(1)
        else:
            aromaticities.append(0)
        temp.clear()
    return aromaticities





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

    #calculate the MorgenFingerprint(ecfp6) as Vector for the molocules.
    fp = []
    for mol in mols:
        fp.append(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=3, nBits = 4096))

    Data["SMILES"] =smi
    Data["Fingerprints"] = fp
    Data.info()

    #calculating logP values vor every molecule in the file.
    total , logp_score_per_molecule = check_logp(Data)


    print(ceck_aromaticity(Data))
    print(len(ceck_aromaticity(Data)))



    #first I want to split the data into a train a validation and a test set with the sklearn packege.
    #one of the common ratios for splitting the data.
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    #because the package can only split between test and train I used the function twice.
    x_train, x_test, y_train, y_test = train_test_split(Data, Data, test_size= 1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size= test_ratio / (test_ratio + validation_ratio))

    #print(x_train, x_test, y_train, y_test)


    return Data

# silmple model for pytorch to optimize the network for our purpose.
class Neural_Network(nn.Module):

    def __init__(self, ):
        super(Neural_Network, self).__init__()

        # parameters
        self.inputSize = 4096
        self.outputSize = 1
        self.hiddenSize = 4096

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
