import torch
import numpy as np
from torch import nn
from data_loader import Data_Loader, Test_Data_Loader
from test import test_on_data
from loss import generalized_cross_entropy_binary, constrained_loss
from layers import FeedforwardNeuralNetModel, LogisticRegression, Learn_from_Rules


data_feeder = Data_Loader(L_processed_file = 'data/YouTube-Spam-Collection-v1/L_preprocess.npy',
                        U_processed_file = 'data/YouTube-Spam-Collection-v1/U_preprocess.npy',
                        batch_size = 16)
test_data_feeder = Test_Data_Loader(test_processed_file = 'data/YouTube-Spam-Collection-v1/test_preprocess.npy',
                                    batch_size=16)

only_l_model = LogisticRegression(16634, 2)
rule_model = FeedforwardNeuralNetModel(16643, [32], 1)
joint_model = Learn_from_Rules(only_l_model,
                            rule_model,
                            16634, 9, 2)

EPOCHS = 60
gamma = 1

criteria1 = nn.CrossEntropyLoss()
criteria2 = nn.BCELoss()
criteria3 = generalized_cross_entropy_binary
criteria4 = constrained_loss
optimizer = torch.optim.Adam(joint_model.parameters(), lr=0.01)

for epoch in range(EPOCHS):
    total_batches = max(data_feeder.get_total_no_batches('only_l'), data_feeder.get_total_no_batches('U'))
    joint_model.train(True)
    training_loss = 0
    for batch_id in range(total_batches):
        optimizer.zero_grad()
        batch_only_l, true_labels, batch_over_generalized, batch_xent = data_feeder.get_batch_from_L()
        batch_U_feat_rule, batch_U_feat, batch_U_rule = data_feeder.get_batch_from_U()

        # print('batch_only_l ',batch_only_l.shape)
        # print('true_labels', true_labels.shape)
        # print('batch_over_generalized', batch_over_generalized.shape)
        # print('batch_xent', batch_xent.shape)
        # print('batch_U_feat_rule', batch_U_feat_rule.shape)
        # print('batch_U_feat', batch_U_feat.shape)
        # print('batch_U_rule', batch_U_rule.shape)


        only_l_output = joint_model.forward_only_L(batch_only_l)
        rule_model_output = joint_model.forward_rule_model(batch_over_generalized)

        U_only_l_output = joint_model.forward_only_L(batch_U_feat)
        U_rule_model_output = joint_model.forward_rule_model(batch_U_feat_rule)
        pt = U_only_l_output[np.arange(U_only_l_output.shape[0]),batch_U_rule]
        # print(only_l_output.shape)
        # print(rule_model_output.shape)
        # print(true_labels.shape)

        loss1 = criteria1(only_l_output, true_labels)
        loss2 = criteria2(rule_model_output , torch.zeros(rule_model_output.shape[0],1))
        loss3 = criteria3(rule_model_output)
        loss4 = criteria4(U_rule_model_output, pt)
        loss = loss1 + loss2 + loss3 - (gamma*loss4)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        # print(loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss.item())
        # break
    print("EPOCH "+str(epoch)+" COMPLETED")
    joint_model.train(False)
    print("TRAINING LOSS : "+str(training_loss/total_batches))
    test_on_data(joint_model, test_data_feeder)