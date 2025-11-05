import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

def _create_sequences(data, seq_length, pred_idx):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, pred_idx] # Predict only the first column ('cumulative_confirmed')
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def clean_data(df, start_date, end_date):
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    clean_df = df.drop(['new_confirmed', 'new_deceased', 'new_recovered',
                        'new_tested', "cumulative_deceased", "cumulative_recovered", 
                        "cumulative_tested"], axis = 1)
    return clean_df
    
def csv_reader(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("Error: Please make sure the file is in the correct directory.")
    except Exception as e:
        print("An error occurred:", e)
    return df

def data_loader(selected_contry_codes, seq_length = 30):
    # Load data
    path = 'data/epidemiology.csv'
    df = csv_reader(path)

    # Clean data
    start_date = '2020-01-06'
    end_date = '2022-06-06'
    clean_df = clean_data(df, start_date,end_date)

    country_dfs = []
    for code in selected_contry_codes:
        country_df = clean_df[clean_df['location_key'] == code].copy()
        country_dfs.append(country_df)

    ####### dataset part #####
    data_list = []
    for i in range(len(selected_contry_codes)):

        # Select the columns for the time series
        cumulative_confirmed_data = country_dfs[i]['cumulative_confirmed'].values.astype(float)

        # Scale the 'cumulative_confirmed' data
        scaler_confirmed = MinMaxScaler(feature_range=(-1, 1))
        cumulative_confirmed_scaled = scaler_confirmed.fit_transform(cumulative_confirmed_data.reshape(-1, 1))

        # X, y = create_sequences(data_combined, seq_length,0)
        X, y = _create_sequences(cumulative_confirmed_scaled, seq_length,0) # perdict temp 0

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Reshape y to have a size of (batch_size, 1)
        y = y.unsqueeze(1)

        # Split into training and testing sets
        train_size = int(len(X) * 0.8)
        test_size = len(X) - train_size

        X_raw_train, X_raw_test = X[:train_size], X[train_size:]
        y_raw_train, y_raw_test = y[:train_size], y[train_size:]

        data_list.append((X_raw_train, y_raw_train, X_raw_test, y_raw_test))

    return data_list