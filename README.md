# An artificial neural network to predict reactivity ratios in radical copolymerization
This repository contains a model for polymer informatics. 

This repository contains a model which predicts reactivity ratios between monomer pairs in a radical copolymerization solely based on the chemical structures of them. More details are described in [our paper](https://www).</sup>

## Instructions for use
To make predictions use "ANN_Prediction.py" file. It uses the "ANN_model.h5" file to get all the weights of the network and then predicts the reactivity ratio values based on the given inputs. Entries must be in the form of the SMILES<sup>*</sup> string. The training code is also provided in which the features are generated based on fingerprints and the model is trained based on them. All the data can be found in *Polymer Handbook; Brandrup, J.; Immergut, E. H.; Grulke, E. A., Eds.; 4th editio.; Wiley, 1999.* (John Wiley & Sons, Inc. holds the copyright of this database).

<sup>*The simplified molecular-input line-entry system (SMILES) is a specification in the form of a line notation for describing the structure of chemical species using short ASCII strings. You can find the SMILES string of a monomer by entering its "Structure Identifier" (such as name, CAS number, etc.) and selecting the "convert to" option as "SMILES", then clicking on submit, at [Here](https://cactus.nci.nih.gov/chemical/structure).</sup>

## Packages
Required packages and thier versions are listed in "requirements.txt".

## Reference

## LICENSE
