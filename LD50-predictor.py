#install all packages for the predictor

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools #
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



SDFFile = "/Users/raphael/Documents/Studium/Vorlesungen/Cheminfo/cheminformatics/qspr-dataset-02.sdf 2"
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

    #one of the common ratios for splitting the data.
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15


    x_train, x_test, y_train, y_test = train_test_split(Data, Data, test_size= 1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size= test_ratio / (test_ratio + validation_ratio))

    print(x_train, x_val, x_test)

    return Data



#start of the funktions
def main():
    datapreprossessing()



#start of the program
if __name__ == "__main__":
    main()
