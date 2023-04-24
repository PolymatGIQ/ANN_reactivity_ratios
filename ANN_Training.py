### Training the ANN
### ver: 3.1
### April 24, 2023

import config
import numpy as np
import pandas as pd

from rdkit import Chem
from mfingerprint import fingerprint

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# Importing dataset
df = pd.read_excel('./RRDataset_Main.xlsx')

# Custom elimination
df = df.loc[(df['r1'] >= config.r_min) & (df['r2'] >= config.r_min)]
df = df.loc[(df['r1'] <= config.r_max) & (df['r2'] <= config.r_max)]
df = df.reset_index()
print('Number of rows:', len(df))

# Transforming to Ln
df.r1 = np.log(df.r1)
df.r2 = np.log(df.r2)

# Generating FPs
nbits = config.bits
Mat1 = [Chem.MolFromSmiles(i) for i in df['S1']]
Mat2 = [Chem.MolFromSmiles(i) for i in df['S2']]
FP1 = [fingerprint(j, nbits) for j in Mat1]
FP2 = [fingerprint(j, nbits) for j in Mat2]

# Adding Names, SMILES, r values to DF
dfm = pd.DataFrame()
dfm.loc[:, 'M1'], dfm.loc[:, 'M2'] = df.M1, df.M2
dfm.loc[:, 'r1'], dfm.loc[:, 'r2'] = df.r1, df.r2

# Combining FPs
df1 = pd.DataFrame(data=FP1, columns=range(1, nbits + 1))
df2 = pd.DataFrame(data=FP2, columns=range(nbits + 1, 2 * nbits + 1))
dff = pd.concat([dfm, df1, df2], axis=1)

# Assigning X and y
y = dff[dff.columns[0:4]].values
X = dff[dff.columns[6::]].values

# Splitting data to Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
print('\nTraining size:', len(X_train))
print('Testing size:', len(X_test))

df_train = pd.DataFrame(data=[i[0:2] for i in y_train], columns=['M1', 'M2'])
df_test = pd.DataFrame(data=[i[0:2] for i in y_test], columns=['M1', 'M2'])
y_train = tf.convert_to_tensor([i[2:4] for i in y_train], dtype=tf.float32)
y_test = tf.convert_to_tensor([i[2:4] for i in y_test], dtype=tf.float32)

# ANN model based on GS_CV
model = Sequential()
model.add(Dense(80, input_dim=X_train.shape[1], kernel_initializer='random_normal', activation='relu'))
model.add(Dense(40, kernel_initializer='random_normal', activation='relu'))
model.add(Dense(2, activation='linear'))
model.summary()

model.compile(loss='mae', optimizer='adam')

monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=config.delta,
    patience=config.patience,
    verbose=1,
    mode='auto',
    restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=config.batches, callbacks=[monitor], epochs=config.epochs, verbose=1)

print('\nFinal Evaluation:', model.evaluate(X_train, y_train, verbose=0))


# Predicting based on the features
y_pred_train = model.predict(X_train, verbose=0)
y_pred_test = model.predict(X_test, verbose=0)

print('Train: R_sqrd= %.2f' % r2_score(y_train, y_pred_train), ' | ',
      'MAE= %.2f' % mean_absolute_error(y_train, y_pred_train))
print('Tests: R_sqrd= %.2f' % r2_score(y_test, y_pred_test), ' | ',
      'MAE= %.2f' % mean_absolute_error(y_test, y_pred_test))

# Back-transformation from Ln
y_test_R = np.exp(y_test)
y_train_R = np.exp(y_train)
y_pred_test_R = np.exp(y_pred_test)
y_pred_train_R = np.exp(y_pred_train)

df_train['r1'], df_train['r2'] = y_train_R[:, 0], y_train_R[:, 1]
df_train['Pr1'], df_train['Pr2'] = y_pred_train_R[:, 0], y_pred_train_R[:, 1]
df_test['r1'], df_test['r2'] = y_test_R[:, 0], y_test_R[:, 1]
df_test['Pr1'], df_test['Pr2'] = y_pred_test_R[:, 0], y_pred_test_R[:, 1]

# Randoms for Train and Test
nors1, nors2 = 450, 50
dfTr = df_train.sample(n=nors1, random_state=1)
dfTe = df_test.sample(n=nors2, random_state=1)

# Producing the figures
figure(dpi=config.DPI)
plt.rcParams['font.size'] = config.font_s
plt.xlim(config.lim), plt.ylim(config.lim)
plt.xlabel('Real r1'), plt.ylabel('Predicted r1')
plt.xticks(config.tick), plt.yticks(config.tick), plt.axis('square')
plt.grid(color=config.grc, linestyle=config.grl, lw=config.grw, alpha=config.alph)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, color=config.pac, linestyle=config.pal, lw=config.paw, alpha=config.alph)
plt.scatter(dfTr['r1'], dfTr['Pr1'], color=config.Tr_c, alpha=config.alph, marker='.', label='train')
plt.scatter(dfTe['r1'], dfTe['Pr1'], color=config.Te_c, alpha=config.alph, marker='.', label='test')
plt.legend(loc='best')
plt.savefig("Figure3_1.TIFF", bbox_inches='tight')
plt.show()

figure(dpi=config.DPI)
plt.rcParams['font.size'] = config.font_s
plt.xlim(config.lim), plt.ylim(config.lim)
plt.xlabel('Real r2'), plt.ylabel('Predicted r2')
plt.xticks(config.tick), plt.yticks(config.tick), plt.axis('square')
plt.grid(color=config.grc, linestyle=config.grl, lw=config.grw, alpha=config.alph)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, color=config.pac, linestyle=config.pal, lw=config.paw, alpha=config.alph)
plt.scatter(dfTr['r2'], dfTr['Pr2'], color=config.Tr_c, alpha=config.alph, marker='.', label='train')
plt.scatter(dfTe['r2'], dfTe['Pr2'], color=config.Te_c, alpha=config.alph, marker='.', label='test')
plt.legend(loc='best')
plt.savefig("Figure3_2.TIFF", bbox_inches='tight')
plt.show()

# Mean Relative Error Calculation
MRE1 = 100 * abs((df_test['r1'] - df_test['Pr1']) / (df_test['r1']))
MRE2 = 100 * abs((df_test['r2'] - df_test['Pr2']) / (df_test['r2']))
df_test['MRE1'], df_test['MRE2'], MRET = MRE1, MRE2, (MRE1 + MRE2) / 2

# Mean Absolute Error Calculation
MAE1 = abs(df_test['r1'] - df_test['Pr1'])
MAE2 = abs(df_test['r2'] - df_test['Pr2'])
df_test['MAE1'], df_test['MAE2'], MAET = MAE1, MAE2, (MAE1 + MAE2) / 2

# Mean Squared Error Calculation
MSE1 = (df_test['r1'] - df_test['Pr1'])**2
MSE2 = (df_test['r2'] - df_test['Pr2'])**2
df_test['MSE1'], df_test['MSE2'], MSET = MSE1, MSE2, (MSE1 + MSE2) / 2

print('\n# The Accuracy of the model#')
print('Mean Relative Error',
      'r1: ', '{:.1f}'.format(abs(np.mean(MRE1))), '%', ' | ',
      'r2: ', '{:.1f}'.format(abs(np.mean(MRE2))), '%', ' | ',
      'Tot:', '{:.1f}'.format(abs(np.mean(MRET))), '%')

print('Mean Absolute Error',
      'r1: ', '{:.2f}'.format(abs(np.mean(MAE1))), ' | ',
      'r2: ', '{:.2f}'.format(abs(np.mean(MAE2))), ' | ',
      'Tot:', '{:.2f}'.format(abs(np.mean(MAET))))

print('Mean Squared Error',
      'r1: ', '{:.2f}'.format(abs(np.mean(MSE1))), ' | ',
      'r2: ', '{:.2f}'.format(abs(np.mean(MSE2))), ' | ',
      'Tot:', '{:.2f}'.format(abs(np.mean(MSET))), '\n')


# Saving test result
df_test.to_csv("ANN_Train.csv", index=False)

# Saving model
model.save("ANN_Train.h5")
print('The model is successfully saved')
