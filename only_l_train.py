import torch
import numpy as np
from torch import nn
from data_loader import Data_Loader_Only_L
from test import test_only_l_on_data
from layers import  LogisticRegression


data_feeder = Data_Loader_Only_L(L_processed_file = 'data/YouTube-Spam-Collection-v1/L_preprocess.npy',
                        batch_size = 16)

test_data_feeder = Data_Loader_Only_L(L_processed_file = 'data/YouTube-Spam-Collection-v1/test_preprocess.npy',
                            batch_size = 16)

only_l_model = LogisticRegression(16634, 2)

EPOCHS = 60
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(only_l_model.parameters(), lr=0.003)


for epoch in range(EPOCHS):
    total_batches = data_feeder.get_total_no_batches()
    training_loss = 0
    only_l_model.train(True)
    for batch_id in range(total_batches):
        feats, true_labels = data_feeder.get_batch()
        output = only_l_model(feats)
        loss = criteria(output, true_labels.long())
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    print("EPOCH "+str(epoch)+" COMPLETED")
    print("TRAINING LOSS : "+str(training_loss/total_batches))
    only_l_model.train(False)
    test_only_l_on_data(only_l_model, test_data_feeder)