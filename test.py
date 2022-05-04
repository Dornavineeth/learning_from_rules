import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def get_G_set_indices(rule_model_output):
    indices = np.where(rule_model_output > 0.5)[0]
    return indices

def test_on_data(joint_model, test_loader):
    total_no_batches = test_loader.get_total_no_batches()
    b_size = test_loader.batch_size

    num_rules = test_loader.test_rule_labels.shape[1]
    y_hat = np.array([])
    y_true = np.array([])
    for batch_id in range(total_no_batches):
        zero_score = np.array([])
        one_score = np.array([])
        batch_only_l, feat_labels, rule_labels = test_loader.get_batch()
        only_l_output = joint_model.forward_only_L(batch_only_l)
        y_true = np.append(y_true, feat_labels)
        for idx in range(only_l_output.shape[0]):
            inst_only_l = batch_only_l[idx]
            inst_feat_labels = feat_labels[idx]
            inst_rule_labels = rule_labels[idx]
            b_only_l = torch.tile(inst_only_l,(num_rules,1))
            inst = torch.eye(num_rules, num_rules)
            rule_feats = torch.cat((b_only_l, inst), axis=1)
            rule_model_output = joint_model.forward_rule_model(rule_feats)
            g_indices = get_G_set_indices(rule_model_output)
            soft_score_0 = torch.tensor(0)
            soft_score_1 = torch.tensor(0)
            for rg in g_indices:
                t1 = rule_model_output[rg][0]*long(inst_rule_labels[rg]==0)
                t1 += (1- rule_model_output[rg][0])*long(inst_rule_labels[rg]!=0)
                t2 = rule_model_output[rg][0]*long(inst_rule_labels[rg]==1)
                t2 += (1- rule_model_output[rg][0])*long(inst_rule_labels[rg]!=1)
                soft_score_0 += t1
                soft_score_1 += t2 
            if len(g_indices)>0:
                soft_score_0 = soft_score_0/len(g_indices)
                soft_score_1 = soft_score_1/len(g_indices)
            
            zero_score = np.append(zero_score, only_l_output[idx][0].item()+soft_score_0.item())
            one_score = np.append(one_score, only_l_output[idx][1].item()+soft_score_1.item())
        y_hat = np.append(y_hat, (zero_score<one_score))

    acc = accuracy_score(y_true, y_hat)
    micro_f1 = f1_score(y_true, y_hat, average='micro')
    macro_f1 = f1_score(y_true, y_hat, average='macro')
    print("ACCURACY ON TEST SET : "+str(acc))
    print("MICRO_F1 ON TEST SET : "+str(micro_f1))
    print("MACRO_F1 ON TEST SET : "+str(macro_f1))

def test_only_l_on_data(model, data_feeder):
    total_batches = data_feeder.get_total_no_batches()
    y_hat = np.array([])
    y_true = np.array([])
    for batch_id in range(total_batches):
        feats, true_labels = data_feeder.get_batch()
        output = model(feats)
        output = output.detach().numpy()
        # print(int(output[:,0] > output[:,1]))
        yout = (output[:,0] < output[:,1])
        yout = np.array([int(x) for x in yout])
        y_hat = np.append(y_hat, yout)
        y_true = np.append(y_true, true_labels)
    
    
    acc = accuracy_score(y_true, y_hat)
    micro_f1 = f1_score(y_true, y_hat, average='micro')
    macro_f1 = f1_score(y_true, y_hat, average='macro')
    print("ACCURACY ON TEST SET : "+str(acc))
    print("MICRO_F1 ON TEST SET : "+str(micro_f1))
    print("MACRO_F1 ON TEST SET : "+str(macro_f1))

