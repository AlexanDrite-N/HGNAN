import torch_geometric as pyg
from get_hypergraph import *

import os
import torch

def get_data(data_name, processed_data_dir='processed_data', model_name='HGNAM', train_size=0.5, val_size=0.25, batch_size = 32):
    print(f'Loading {data_name} dataset')
    train_loader, val_loader, test_loader = None, None, None
    num_features = None
    
    if not os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
        if model_name == 'HGNAM':
          if data_name in ['cora']:
              data = load_citation_dataset(data_name=data_name, train_size=train_size, val_size=val_size)
          else:
              data = load_LE_dataset(data_name=data_name, train_size=train_size, val_size=val_size)
        else:
            if data_name in ['congress-bills', 'contact-high-school', 'email-Enron', 'NDC-classes']:
                data = get_hypergraph_with_noise(dataset=data_name, train_size=train_size, feature_noise=0.5)
            else:
                data = get_hypergraph(dataset=data_name, train_size=train_size)
    else:
        data = torch.load(f'{processed_data_dir}/{data_name}.pt', weights_only=False)
        print(f'Loaded {data_name} dataset')
    print("Preprocess Finished!")

    if model_name == 'HGNAM':
      if data_name == 'cora':
          num_classes = 7
      elif data_name == 'Mushroom':
          num_classes = 2
      elif data_name == 'NTU2012':
          num_classes = 67
      elif data_name == 'zoo':
          num_classes = 7
    else:
        num_classes = 2
    
    if train_loader is None:
        train_data = data
        val_data = data
        test_data = data
        train_loader = pyg.loader.DataLoader([train_data], batch_size=batch_size)
        val_loader = pyg.loader.DataLoader([val_data], batch_size=batch_size)
        test_loader = pyg.loader.DataLoader([test_data], batch_size=batch_size)

    if num_features is None:
        num_features = train_data.x.shape[1]

    return train_loader, val_loader, test_loader, num_features, num_classes

