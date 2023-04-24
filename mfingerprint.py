import numpy as np
from rdkit.Chem import AllChem, DataStructs


def fingerprint(mol, nbits):
    npfp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nbits), npfp)
    return npfp

