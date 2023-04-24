### Prediction
### ver: 3.1
### April 24, 2023

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tensorflow.keras.models import load_model

# Importing the model
model = load_model('./ANN_model.h5')
model.summary()

# Getting the strings of monomers
User_M1 = input("Enter SMILES for 1st monomer: ")
User_M2 = input("Enter SMILES for 2nd monomer: ")

# Generating Fingerprints
User_1, User_2 = [], []
Mat1 = Chem.MolFromSmiles(str(User_M1))
Mat2 = Chem.MolFromSmiles(str(User_M2))
FP1 = AllChem.GetMorganFingerprintAsBitVect(Mat1, radius=3, nBits=2048)
FP2 = AllChem.GetMorganFingerprintAsBitVect(Mat2, radius=3, nBits=2048)

# Combining FPs 
FPss_1 = np.concatenate((FP1, FP2), axis=0)
FPss_2 = np.concatenate((FP2, FP1), axis=0)
User_1.append(FPss_1)
User_2.append(FPss_2)

# Predicting based on the features
FPL_1, FPL_2 = np.array(User_1), np.array(User_2)
pred_User_1 = model.predict(FPL_1[0:], verbose=0)
pred_User_R_1 = np.exp(pred_User_1)
pred_User_2 = model.predict(FPL_2[0:], verbose=0)
pred_User_R_2 = np.exp(pred_User_2)

# Printing the results
Final1 = round(((pred_User_R_1[0][0] + pred_User_R_2[0][1]) / 2), 3)
Final2 = round(((pred_User_R_2[0][0] + pred_User_R_1[0][1]) / 2), 3)
print(Final1, Final2)
