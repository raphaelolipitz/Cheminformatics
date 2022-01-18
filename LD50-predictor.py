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
from keras.models import Sequential
from keras.layers import Dense

import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.jit import script, trace       # hybrid frontend decorator and tracing jit



SDFFile = "/Users/raphael/Downloads/qspr-dataset-02.sdf"
Data = PandasTools.LoadSDF(SDFFile,smilesName='SMILES',
                           molColName='Molecule',
                           includeFingerprints=True,
                           removeHs=False, strictParsing=True)

Data.info()

smi = []
mols = [mol for mol in Chem.SDMolSupplier(SDFFile)]

for mol in mols:
    smi.append(Chem.MolToSmiles(mol))

fp = []
for mol in mols:
    fp.append(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits = 124))


def function():
    return

#start of the program
def main():
    function()
if __name__ == "__main__":
    main()
