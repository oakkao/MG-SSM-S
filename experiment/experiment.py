import torch
import time
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from src.utils.data_loader import data_loader
from scripts.train import ModelTraining

# Data preparation
## Select Data
selected_contry_codes = ['US', 'IN', 'BR', 'FR', 'DE',
                        'GB', 'RU', 'IT', 'TR', 'ES',
                        'VN', 'AR', 'AU', 'AT', 'BD',
                        'BE', 'BG', 'CA', 'CL', 'CN', 
                        'CU', 'DK', 'FI', 'GE', 'GR',
                        'ID', 'JP', 'JO', 'KE', 'KR',
                        'LR', 'MY', 'ML', 'MX', 'NL', 
                        'NO', 'PH', 'SE', 'CH', 'TH']

data_list = data_loader(selected_contry_codes, 30)

# Set a fixed seed for reproducibility
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

import src.model as md

model_config_list = [(md.baseline_models.OriginalLSTMModel,
                {"input_size":1,
                "hidden_size":256,
                "num_layers":1,
                "output_size":1}),
                # (md.baseline_models.BidirectionalLSTMModel,
                # {"input_size":1,
                # "hidden_size":256,
                # "num_layers":1,
                # "output_size":1}),
                # (md.baseline_models.UnidirectionalGRUModel,
                # {"input_size":1,
                # "hidden_size":128,
                # "num_layers":1,
                # "output_size":1}),
                # (md.custom_mamba.CustomMambaModel,
                # {"input_size":1,
                # "hidden_size":32,
                # "num_layers":1,
                # "output_size":1}),
                (md.mg_smm.MgSmmModel,
                {"input_size":1,
                "hidden_size":64,
                "num_layers":1,
                "output_size":1,
                "gate_size":32}),
                (md.mg_smm_s.MgSmmSModel,
                {"input_size":1,
                "hidden_size":128,
                "num_layers":1,
                "output_size":1,
                "gate_size":64}),
                ]

# parameter for training
num_epochs = 1000
learning_rate = 0.001
batch_size = 64

# parameter for experiments
num_runs = 1
start = 0
stop = 40

All_result = []

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")

for i in range(len(data_list[start:stop])):

  idx = start + i
  data = data_list[idx]
  print(selected_contry_codes[idx])

  X_raw_train, y_raw_train, X_raw_test, y_raw_test = data

  eval_size = int(len(X_raw_test) * 0.5)
  X_raw_eval, X_raw_test =  X_raw_test[:eval_size], X_raw_test[eval_size:]
  y_raw_eval, y_raw_test = y_raw_test[:eval_size], y_raw_test[eval_size:]

  country_run = {"country_code":selected_contry_codes[start + i]}

  train_data = TensorDataset(X_raw_train, y_raw_train)
  eval_data = TensorDataset(X_raw_eval, y_raw_eval)
  test_data = TensorDataset(X_raw_test, y_raw_test)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
  eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

  for model_config in model_config_list:
        model, config = model_config
        start_time = time.time()
        test_losses =  ModelTraining(model_config, train_loader, eval_loader, test_loader, num_epochs, learning_rate, num_runs)
        end_time = time.time()
        elasped_time = end_time - start_time

        country_run[model.__name__ + '_test_losses'] = test_losses
        country_run[model.__name__ + '_test_losses_time'] = elasped_time

        All_result.append(country_run)

results_df = pd.DataFrame(All_result)#.T
results_df.to_csv(f'results.csv')