import torch
import torch.nn as nn

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class FeedforwardNeuralNetModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        
        self.layers = []
        self.layers.append(nn.Linear(input_dim,hidden_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(1,len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dims[-1],output_dim))
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.model(x)


class Learn_from_Rules(nn.Module):

    def __init__(self, 
                only_L_model,
                rule_model,
                num_features, 
                num_rules,
                num_classes):
        super(Learn_from_Rules, self).__init__()

        self.only_L_model = only_L_model
        self.rule_model = rule_model
        self.num_features = num_features
        self.num_rules = num_rules
        self.num_classes = num_classes
        
    def forward_only_L(self, batch):
        output = self.only_L_model(batch)
        # print(output.shape)
        return output

    def forward_rule_model(self, batch):
        output = self.rule_model(batch)
        return output


    