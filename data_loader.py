import torch
import numpy as np

class Data_Loader():

    def __init__(self, L_processed_file,
                    U_processed_file,
                    batch_size = 16):
        with open(L_processed_file,'rb') as f:
            self.L_feats = np.load(f, allow_pickle=True)
            self.L_rule_labels = np.load(f, allow_pickle=True)
            self.L_coverage = np.load(f, allow_pickle=True)
            self.L_feats_label = np.load(f, allow_pickle=True)
            self.L_r = np.load(f, allow_pickle=True)
        
        with open(U_processed_file,'rb') as f:
            self.U_feats = np.load(f, allow_pickle=True)
            self.U_rule_labels = np.load(f, allow_pickle=True)
            self.U_coverage = np.load(f, allow_pickle=True)
            self.U_feats_label = np.load(f, allow_pickle=True)
            self.U_r = np.load(f, allow_pickle=True)
        
        
        self.batch_size = batch_size
        self.batch_counter = {'only_l':0,
                             'U':0}

        self.covered_U_mask = self.get_covered_mask_U()
        self.covered_U_indices = self.get_covered_indices_U()
        self.covered_U_feats = self.U_feats[self.covered_U_mask]
        self.covered_U_rule_labels = self.U_rule_labels[self.covered_U_mask]
        self.covered_U_coverage = self.U_coverage[self.covered_U_mask]
        self.covered_U_feats_label = self.U_feats_label[self.covered_U_mask]
        self.covered_U_r = self.U_r[self.covered_U_mask]
        

    def get_total_no_batches(self, mode='only_l'):
        if mode == 'only_l':
            if self.L_feats.shape[0] % self.batch_size == 0:
                return self.L_feats.shape[0] // self.batch_size

            else:
                return (self.L_feats.shape[0] // self.batch_size)+1
        if mode == 'U':
            if len(self.covered_U_indices) % self.batch_size == 0:
                return len(self.covered_U_indices) // self.batch_size

            else:
                return (len(self.covered_U_indices) // self.batch_size)+1


    def get_covered_mask_U(self):
        mask = np.array(np.sum(self.U_coverage,axis=1) > 0)
        return mask
    
    def get_covered_indices_U(self):
        indices = np.where(np.sum(self.U_coverage,axis=1) > 0)[0]
        return indices

    # start = self.batch_counter['U']*self.batch_size
    #     end = min(self.batch_counter['U']*self.batch_size + self.batch_size, len(self.covered_U_indices))
        
    def get_batch_U(self, start, end):
        num_rules = self.covered_U_rule_labels.shape[1]
        batch_U = self.covered_U_rule_labels[start:end]
        true_labels = self.covered_U_feats_label[start:end]
        feat_rule_batch = np.zeros((0, num_rules+self.U_feats.shape[1]))
        feat_batch = np.zeros((0, self.covered_U_feats.shape[1]))
        rule_batch = np.array([])
        for i in range(0,end-start):
            rule_id = np.where(batch_U[i] !=-1)[0]
            if len(rule_id)==0:
                continue
            b = np.zeros((len(rule_id), num_rules))
            b[np.arange(len(rule_id)), rule_id] = 1
            feats = np.tile(self.covered_U_feats[i+start],(len(rule_id),1))
            inst = np.concatenate((feats,b), axis=1)
            # print(feat_rule_batch.shape, inst.shape)
            feat_rule_batch = np.concatenate((feat_rule_batch, inst), axis=0)
            feat_batch = np.concatenate((feat_batch, feats), axis=0)
            rule_batch = np.append(rule_batch, batch_U[i][rule_id])
        return torch.from_numpy(np.array(feat_rule_batch)).float(), torch.from_numpy(np.array(feat_batch)).float(), np.array(rule_batch)
    
    def get_batch_from_U(self):
        start = self.batch_counter['U']*self.batch_size
        end = min(self.batch_counter['U']*self.batch_size + self.batch_size, self.covered_U_feats.shape[0])
        if end == self.covered_U_feats.shape[0]:
            self.batch_counter['U'] = 0
        else:
            self.batch_counter['U'] += 1
        return self.get_batch_U(start, end)

    def get_batch_only_L(self,start,end):
        return self.L_feats[start:end] , self.L_feats_label[start:end]  

    def get_batch_xent(self,start,end):
        num_rules = self.L_rule_labels.shape[1]
        batch_L = self.L_rule_labels[start:end]
        true_labels = self.L_feats_label[start:end]
        final_batch = np.zeros((0,num_rules+self.L_feats.shape[1]))

        for i in range(0,end-start):
            rule_id = np.where((batch_L[i] !=-1) & (batch_L[i]==true_labels[i]))[0]
            if len(rule_id)==0:
                continue
            b = np.zeros((len(rule_id),num_rules))
            b[np.arange(len(rule_id)), rule_id] = 1
            feats = np.tile(self.L_feats[i+start],(len(rule_id),1))
            inst = np.concatenate((feats,b),axis=1)
            final_batch = np.concatenate((final_batch, inst),axis=0)
        return np.array(final_batch)

    def get_batch_over_generalized(self,start,end):
        num_rules = self.L_rule_labels.shape[1]
        batch_L = self.L_rule_labels[start:end]
        true_labels = self.L_feats_label[start:end]
        final_batch = np.zeros((0,num_rules+self.L_feats.shape[1]))

        for i in range(0,end-start):
            rule_id = np.where((batch_L[i] !=-1) & (batch_L[i]!=true_labels[i]))[0]
            if len(rule_id)==0:
                continue
            b = np.zeros((len(rule_id),num_rules))
            b[np.arange(len(rule_id)), rule_id] = 1
            feats = np.tile(self.L_feats[i+start],(len(rule_id),1))
            inst = np.concatenate((feats,b),axis=1)
            final_batch = np.concatenate((final_batch, inst),axis=0)
        return np.array(final_batch)
    
    def get_batch_from_L(self):
        start = self.batch_counter['only_l']*self.batch_size
        end = min(self.batch_counter['only_l']*self.batch_size + self.batch_size, self.L_feats.shape[0])
        
        batch_only_l, true_labels = self.get_batch_only_L(start,end)
        batch_over_generalized = self.get_batch_over_generalized(start,end)
        batch_xent = self.get_batch_xent(start,end)

        if end == self.L_feats.shape[0]:
            self.batch_counter['only_l'] = 0
        else:
            self.batch_counter['only_l'] += 1

        batch_only_l = torch.from_numpy(batch_only_l).float()
        true_labels = torch.from_numpy(true_labels).long()
        batch_over_generalized = torch.from_numpy(batch_over_generalized).float()
        batch_xent = torch.from_numpy(batch_xent).float()

        return batch_only_l, true_labels, batch_over_generalized, batch_xent
    
class Test_Data_Loader():
    def __init__(self, test_processed_file, batch_size = 16):
        with open(test_processed_file,'rb') as f:
            self.test_feats = np.load(f, allow_pickle=True)
            self.test_rule_labels = np.load(f, allow_pickle=True)
            self.test_coverage = np.load(f, allow_pickle=True)
            self.test_feats_label = np.load(f, allow_pickle=True)
            self.test_r = np.load(f, allow_pickle=True)
        
         
        self.batch_size = batch_size
        self.num_rules = self.test_rule_labels.shape[0]
        self.batch_counter = 0
    
    def get_total_no_batches(self):
        if self.test_feats.shape[0] % self.batch_size == 0:
            return self.test_feats.shape[0] // self.batch_size
        else:
            return (self.test_feats.shape[0] // self.batch_size)+1

    def get_batch(self):
        start = self.batch_counter * self.batch_size
        end = min(self.batch_counter * self.batch_size + self.batch_size, self.test_feats.shape[0])
        batch_only_l = self.test_feats[start:end]
        feat_labels = self.test_feats_label[start:end]
        rule_labels = self.test_rule_labels[start:end]
        if end == self.test_feats.shape[0]:
            self.batch_counter = 0
        else:
            self.batch_counter += 1
        batch_only_l = torch.from_numpy(batch_only_l).float()
        feat_labels = torch.from_numpy(feat_labels).float()
        rule_labels = torch.from_numpy(rule_labels).float()
        return batch_only_l, feat_labels, rule_labels


class Data_Loader_Only_L():

    def __init__(self, L_processed_file,
                    batch_size = 16):
        with open(L_processed_file,'rb') as f:
            self.L_feats = np.load(f, allow_pickle=True)
            self.L_rule_labels = np.load(f, allow_pickle=True)
            self.L_coverage = np.load(f, allow_pickle=True)
            self.L_feats_label = np.load(f, allow_pickle=True)
            self.L_r = np.load(f, allow_pickle=True)
        
        self.batch_counter = 0
        self.batch_size = batch_size
    
    def get_total_no_batches(self):
        if self.L_feats.shape[0] % self.batch_size == 0:
            return self.L_feats.shape[0] // self.batch_size
        else:
            return (self.L_feats.shape[0] // self.batch_size)+1
    
    def get_batch(self):
        start = self.batch_counter * self.batch_size
        end = min(self.batch_counter * self.batch_size + self.batch_size, self.L_feats.shape[0])
        
        feats = torch.from_numpy(self.L_feats[start:end]).float()
        labels = torch.from_numpy(self.L_feats_label[start:end]).float()

        if end == self.L_feats.shape[0]:
            self.batch_counter = 0
        else:
            self.batch_counter += 1

        return feats, labels 
    
        