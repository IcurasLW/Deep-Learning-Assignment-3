import os
import qlib
from qlib.data import D
from qlib.data.filter import NameDFilter, ExpressionDFilter
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import label_binarize
import csv
from models import *
from datetime import datetime




class ProgressMeter:
    def __init__(self, args, mode, meters:dict, prefix="", ):
        self.meters = meters
        self.prefix = prefix
        self.args = args
        self.mode = mode
        
    def display(self):
        '''
        Will print [Epoch: i] --> loss: loss | Accuracy: acc |  
        '''
        keys = self.meters.keys()
        output = self.prefix
        
        for i in keys:
            output += f' |{i} : {round(self.meters[i], 4)}'

        print(output)
        self.save_results()
    
    def save_results(self):
        '''
        Each row is one epoch
        '''
        output_filename = self.args.save_path + f'{self.mode}_{self.args.modelname}_{self.args.loss}_results.csv'
        # save train results
        
        
        if not os.path.exists(self.args.save_path):
            # If it doesn't exist, create it
            os.mkdir(self.args.save_path)
        
        
        try:
            with open (output_filename, 'r') as log:
                pass
            
            print('file exists')
            with open(output_filename, 'a+', newline='') as log:
                writer = csv.DictWriter(log, fieldnames=self.meters.keys())
                writer.writerow(self.meters)
        except:
            print('file not exists')
            with open(output_filename, 'w', newline='') as log:
                writer = csv.DictWriter(log, fieldnames=self.meters.keys())
                writer.writeheader()
                writer.writerow(self.meters)


def retrieve_data(args, output_path):
    train_output_path = os.path.join(output_path, 'train')
    test_output_path = os.path.join(output_path, 'test')
    
    start_train_time_step = '2010-01-01'
    end_train_time_step = '2017-12-31'
    start_test_time_step = '2018-01-01'
    end_test_time_step = '2019-01-01'
    
    
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')
    ##### Train data #####
    fields = ['$close', '$volume']
    instruments_obj = D.instruments(market='csi300')
    
    train_list_instruments = D.list_instruments(instruments=instruments_obj, 
                                          start_time=start_train_time_step, 
                                          end_time=end_train_time_step, 
                                          as_list=True)
    
    ##### Test data #####
    fields = ['$close', '$volume']
    instruments_obj = D.instruments(market='csi300')
    
    test_list_instruments = D.list_instruments(instruments=instruments_obj, 
                                          start_time=start_test_time_step, 
                                          end_time=end_test_time_step, 
                                          as_list=True)
    
    for instrm in train_list_instruments:
        df_train = D.features([instrm], 
                            fields, 
                            start_time=start_train_time_step, 
                            end_time=end_train_time_step, 
                            freq='day').dropna()
        
        if len(df_train) >= args.window_size:
            df_train = df_train.loc[:, ['$close', '$volume']]
            df_train.to_csv(os.path.join(train_output_path, f'{instrm}.csv'))

    for instrm in test_list_instruments:
        df_test = D.features([instrm], 
                            fields, 
                            start_time=start_test_time_step, 
                            end_time=end_test_time_step, 
                            freq='day').dropna()
        if len(df_test) >= args.window_size:
            df_test = df_test.loc[:, ['$close', '$volume']]
            df_test.to_csv(os.path.join(test_output_path, f'{instrm}.csv'))


def calculate_residual_label(data_path):
    def get_label(x):
        if x == 0:
            return 0
        elif x > 0:
            return 1
        elif x < 0:
            return 2
        
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    train_file = os.listdir(train_path)
    test_file = os.listdir(test_path)
    
    for file in train_file:
        df_train = pd.read_csv(os.path.join(train_path, file))
        price_residual = df_train['$close'][1:].values - df_train['$close'][:-1].values
        volume_residual = df_train['$volume'][1:].values - df_train['$volume'][:-1].values
        price_residual = np.hstack((np.array([0]), price_residual))
        volume_residual = np.hstack((np.array([0]), volume_residual))
        df_train['price_residual'] = pd.Series(price_residual)
        df_train['volume_residual'] = pd.Series(volume_residual)
        df_train['label'] = df_train['price_residual'].map(get_label)
        label = df_train['label'].values[1:]
        df_train = df_train.drop(df_train.index[-1])
        df_train['label'] = label
        df_train.to_csv(os.path.join(train_path, file), index=False)

    for file in test_file:
        df_test = pd.read_csv(os.path.join(test_path, file))
        price_residual = df_test['$close'][1:].values - df_test['$close'][:-1].values
        volume_residual = df_test['$volume'][1:].values - df_test['$volume'][:-1].values
        price_residual = np.hstack((np.array([0]), price_residual))
        volume_residual = np.hstack((np.array([0]), volume_residual))
        df_test['price_residual'] = pd.Series(price_residual)
        df_test['volume_residual'] = pd.Series(volume_residual)
        df_test['label'] = df_test['price_residual'].map(get_label)
        label = df_test['label'].values[1:]
        df_test = df_test.drop(df_test.index[-1])
        df_test['label'] = label
        df_test.to_csv(os.path.join(test_path, file), index=False)



class QlibDataset(Dataset):
    def __init__(self, args, data_path, mean, std):
        self.stock_list = np.array(os.listdir(data_path))
        self.data_path = data_path
        self._mean = mean
        self._std = std
        self.args = args
    
    def __len__(self):
        return len(self.stock_list)
    
    def __getitem__(self, idx):
        data_cols = list(self._mean.keys())
        stock = self.stock_list[idx]
        df_data = pd.read_csv(os.path.join(self.data_path, stock))
        df_data['datetime'] = pd.to_datetime(df_data['datetime'], format = '%Y-%m-%d')
        label = torch.tensor(df_data['label'].to_numpy()[self.args.window_size:])
        time = df_data['datetime'].to_numpy()
        #### normalization
        df_data = df_data[data_cols]
        for col in data_cols:
            df_data[col] = (df_data[col] - self._mean[col]) / self._std[col]
        
        start = 0
        end = start + self.args.window_size
        data = df_data.to_numpy()
        length = data.shape[0]
        output_data = []
        for _ in range(length - self.args.window_size):
            output_data.append(data[start:end, :])
            start += 1
            end = start + self.args.window_size
        
        output_data = torch.tensor(np.array(output_data))
        
        return output_data, stock, time, label


def collate_fn(batch):
    data, stock_name, time, labels = zip(*batch)
    data = pad_sequence(list(data), batch_first=True, padding_value=np.inf)
    labels = pad_sequence(list(labels), batch_first=True, padding_value=-1)
    return data, stock_name, time, labels

def load_data(args, data_path):
    mean, std = calculate_mean_std(os.path.join(data_path, 'train'))
    train_dataset = QlibDataset(args, os.path.join(data_path, 'train'), mean, std)
    test_dataset = QlibDataset(args, os.path.join(data_path, 'test'), mean, std)
    
    shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return train_loader, test_loader

def calculate_mean_std(data_path):
    
    list_instruments = os.listdir(data_path)
    list_data = []
    
    for i in list_instruments:
        df = pd.read_csv(os.path.join(data_path, i))
        df = df.drop(columns=['label','datetime','instrument'])
        columns = df.columns
        list_data.append(df.to_numpy())
        break
    list_data = np.vstack(np.array(list_data))
    mean = list_data.mean(axis=0)
    std = list_data.std(axis=0)
    
    col_mean = {columns[i]:mean[i] for i in range(len(columns))}
    col_std = {columns[i]:std[i] for i in range(len(columns))}
    return col_mean, col_std 

def save_model(val_loss, best_loss, model, args):
    if val_loss <= best_loss:
        torch.save(model.state_dict(), f'best_{args.modelname}_{args.loss}.pth')
    return best_loss


def roc_aupr_score(args, y_true, y_score, average="macro"):
    
    '''
    y_score: the probality of prediction in shape: [num_sample, num_class]
    y_true: the model predcition in shape: [num_sample, num_class], it's one-hot reprensentation of class
    '''
    y_one_hot = label_binarize(y=y_true, classes=[i for i in range(args.num_class)])

    def _binary_roc_aupr_score(y_one_hot, y_score):
        precision, recall, _ = precision_recall_curve(y_one_hot, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_one_hot, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_one_hot, y_score)
        if average == "micro":
            y_one_hot = y_one_hot.ravel()
            y_score = y_score.ravel()
        if y_one_hot.ndim == 1:
            y_one_hot = y_one_hot.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_one_hot.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_one_hot, y_score, average)


def evaluate(args, y_pred, y_true):
    # y_pred = F.softmax(y_pred, dim=1)
    pred_max_indices = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_pred=pred_max_indices, y_true=y_true)
    f1_macro = f1_score(y_pred=pred_max_indices, y_true=y_true, average='macro', zero_division=1)
    precision_macro = precision_score(y_pred=pred_max_indices, y_true=y_true, average='macro', zero_division=1)
    recall_macro = recall_score(y_pred=pred_max_indices, y_true=y_true, average='macro', zero_division=1)
    auc_score_macro = roc_auc_score(y_score=y_pred, y_true=y_true, multi_class='ovr', average='macro')
    aupr_score_macro = roc_aupr_score(args, y_true, y_pred, average='macro')
    
    output = {
            'acc':acc,
            'PR_macro': precision_macro,
            'RE_macro': recall_macro,
            'f1_macro': f1_macro,
            'AUC_macro': auc_score_macro,
            'AUPR_macro': aupr_score_macro
            }
    return output


def prepared_model(args, input_size, hidden_size, num_stacked_layers):
    if args.modelname == 'RNN':
        model = RNN(input_size=input_size,
                hidden_size=hidden_size,
                num_stacked_layers=num_stacked_layers,
                args=args).to(DEVICE)
        
    elif args.modelname == 'GRU':
        model = GRU(input_size=input_size,
                hidden_size=hidden_size,
                num_stacked_layers=num_stacked_layers,
                args=args).to(DEVICE)
        
    elif args.modelname == 'LSTM':
        model = LSTM(input_size=input_size,
                hidden_size=hidden_size,
                num_stacked_layers=num_stacked_layers,
                args=args).to(DEVICE)
        
    elif args.modelname == 'Transformer':
        num_layers = 3
        num_heads = 4
        output_size = 3
        model =  Transformer(args, input_size, hidden_size, num_layers, num_heads, output_size).to(DEVICE)
    return model