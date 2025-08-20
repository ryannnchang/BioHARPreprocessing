import os 
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

input_dir = 'BiomedicHARDataset'
output_dir = '../../data/bioHAR'

data_labels = {"walking":0.0, "walkingup": 1.0, 'walkingdown': 2.0, 'sitting': 3.0, 'standing': 4.0, 'laying': 5.0}

#Helper functions to load data
def window(x, y, z, label):
  X_data_w = []
  y_data_w = []

  start = 0
  window_size = 128
  step_size = 64

  while start + window_size <= len(x):
    window_data = np.array([
      x[start:start+window_size],
      y[start:start+window_size],
      z[start:start+window_size]
        ])
    X_data_w.append(window_data)
    y_data_w.append(label)

    start += step_size

  X_data_w = np.array(X_data_w)
  y_data_w = np.array(y_data_w)

  return X_data_w, y_data_w

def convert(df):
  data_labels = {"walking":0.0, "walkingup": 1.0, 'walkingdown': 2.0, 'sitting': 3.0, 'standing': 4.0, 'laying': 5.0}

  x = pd.DataFrame(df.iloc[:, 1])
  y = pd.DataFrame(df.iloc[:, 2])
  z = pd.DataFrame(df.iloc[:, 3])
  label = data_labels[df.iloc[0, 4]]

  x = np.squeeze(np.array(x))
  y = np.squeeze(np.array(y))
  z = np.squeeze(np.array(z))

  return x, y, z, label

#loading in Ryan's Data
def load_data(data_dir):
  x_data_total = []
  y_data_total = []
  print(f"Loading {data_dir}")

  data = [f for f in os.listdir(data_dir) 
             if not f.startswith('.') and os.path.isdir(os.path.join(data_dir, f))]

  for folder in data:
    files = os.listdir(os.path.join(data_dir, folder))
    files = [f for f in files if not f.startswith('.')]

    for i in files:
      x, y, z, label = convert(pd.read_csv(os.path.join(data_dir, folder, i)))
      x_data, y_data = window(x, y, z, label)
      x_data_total.append(x_data)
      y_data_total.append(y_data)
      print(x_data.shape, y_data.shape)
      #outs (n, 3, 128) and (n, )

    X_all = np.concatenate(x_data_total, axis=0)
    y_all = np.concatenate(y_data_total, axis=0)

  print(X_all.shape, y_all.shape)
  unique, counts = np.unique(y_all, return_counts=True)
  print(dict(zip(unique, counts)))

  return X_all, y_all

X_1, y_1 = load_data(f'{input_dir}/0001')
X_2, y_2 = load_data (f'{input_dir}/0002')
X_3, y_3 = load_data(f'{input_dir}/0003')
X_4, y_4 = load_data(f'{input_dir}/0004')
X_5, y_5 = load_data(f'{input_dir}/0005')

X_12345 = np.concatenate([X_1, X_2, X_3, X_4, X_5], axis=0) 
y_12345 = np.concatenate([y_1, y_2, y_3, y_4, y_5], axis=0) 

X_train, X_30, y_train, y_30 = train_test_split(X_12345, y_12345, train_size=0.7, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_30, y_30, train_size=0.5, random_state=42)

print(X_train.shape, y_train.shape)
print(X_val.shape, X_val.shape)
print(y_test.shape, X_test.shape)

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train)
dat_dict["labels"] = torch.from_numpy(y_train)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val)
dat_dict["labels"] = torch.from_numpy(y_val)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))

print('Uploaded BioHAR Data')