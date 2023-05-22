# An artificial neural network to predict reactivity ratios in radical copolymerization
This repository contains a model which predicts reactivity ratios between monomer pairs in a radical copolymerization solely based on their chemical structures. More details are described in [our paper](https://pubs.rsc.org/en/content/articlelanding/2023/PY/D3PY00246B). The user interface is also available at [PolymatAI](http://polymatai.pythonanywhere.com/search).

## Instruction
To make predictions use *"ANN_Prediction.py"* file. It uses the *"ANN_model.h5"* file to get all the weights of the network and then predicts the reactivity ratio values based on the given inputs. Entries must be in the form of the SMILES<sup>*</sup> string. The training code is also provided in which the features are generated based on fingerprints and the model is trained based on them. All the data can be found in *Polymer Handbook; Brandrup et al.; 4th edition.; Wiley, 1999.* (John Wiley & Sons, Inc. holds the copyright of this database).

<sub>*The simplified molecular-input line-entry system (SMILES) is a specification in the form of a line notation for describing the structure of chemical species using short ASCII strings. You can find the SMILES string of a monomer by entering its "Structure Identifier" (such as name, CAS number, etc.) and selecting the "convert to" option as "SMILES", then clicking on submit, at [Here](https://cactus.nci.nih.gov/chemical/structure).</sub>

## Packages
Required packages and thier versions are listed in the [requirements](https://github.com/PolymatGIQ/ANN_reactivity_ratios/blob/main/requirements.txt).

## LICENSE
This repository is licensed under CC BY-NC 4.0.
For more information please refer to the [license section](https://github.com/PolymatGIQ/ANN_reactivity_ratios/blob/main/LICENSE.md).
