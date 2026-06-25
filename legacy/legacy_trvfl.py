import torch
import torch.nn as nn

class TRVFL(nn.Module):

    def __init__(self, l, features, raw_features, nodes, lamb, input_scale, device, nt_layers=1, num_heads=1,
                 dropout_rate=0.2):
        super(TRVFL, self).__init__()
        
        self.l = l
        self.features = features
        self.raw_features = raw_features
        self.input_scale = input_scale
        self.nodes = nodes
        self.lamb = lamb
        self.d = device
    
        # init params for transformer layer
        self.nt_layers = int(nt_layers)
        self.dropout_rate = dropout_rate
    
        # init layers
        if self.l == 0:
            if self.features%num_heads == 0: self.num_heads = num_heads
            else: self.num_heads = 1
            self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d_model=self.features, 
                nhead=self.num_heads, 
                dim_feedforward=self.nodes, 
                dropout=self.dropout_rate,
                activation='relu', 
                batch_first=True), 
                num_layers=self.nt_layers)
            self.__init_transformer_weights(self.features)
            self.fc1 = nn.Sequential(nn.Linear(in_features=self.features, out_features=self.nodes),
                                     nn.Tanh(),)
        else:
            if (self.features + self.raw_features) == 0: self.num_heads = num_heads
            else: self.num_heads = 1
            self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d_model=self.features + self.raw_features, 
                nhead=self.num_heads, 
                dim_feedforward=self.nodes,
                dropout=self.dropout_rate, 
                activation='relu', 
                batch_first=True), 
                num_layers=self.nt_layers)
            self.__init_transformer_weights(self.features + self.raw_features)
            self.fc1 = nn.Sequential(nn.Linear(in_features=self.features + self.raw_features, out_features=self.nodes),
                                     nn.Tanh(),)
    
        self.fc1.apply(self.__init_weights__)
    
        self.output = nn.Sequential(
            nn.Linear(self.nodes + self.raw_features + 1, 1))
    
    def __init_weights__(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-self.input_scale, self.input_scale)
            m.bias.data.uniform_(-self.input_scale, self.input_scale)
    
    def __init_transformer_weights(self, d_model):
        for name, param in self.transformer.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -self.input_scale, self.input_scale)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def init_weight(self, X, y, X_raw):
        n_sample = X.shape[0]
        encoding = self.transform(X_raw, X)
        merged = torch.cat((encoding, X_raw, torch.ones((n_sample, 1)).to(self.d)), dim=1)  # for direct links
        n_features = merged.shape[1]

        if n_features < n_sample:
            # prime space equation
            # (I.lamb + D^T.D)^-1 . D^T . Y
            beta = torch.mm(
                torch.mm(torch.inverse(torch.eye(merged.shape[1]).to(self.d) * self.lamb + torch.mm(merged.T, merged)),
                         merged.T), y)
        else:
            # dual space equation
            # D^T . (lamb.I + D.D^T)^-1 . Y
            beta = torch.mm(merged.T, torch.mm(
                torch.inverse(torch.eye(merged.shape[0]).to(self.d) * self.lamb + torch.mm(merged, merged.T)), y))
            
        self.output[0].weight = nn.Parameter(beta.T)
        self.output[0].bias.data.fill_(0)

    def transform(self, X_raw, X=None):

        if self.l == 0:
            x = self.transformer(X_raw)
            encoding = self.fc1(x)
        else:
            x = self.transformer(torch.cat((X, X_raw), dim=1))
            encoding = self.fc1(x)
    
        return encoding
    

    def forward(self, X, X_raw):

        if self.l == 0:
            x = self.transformer(X_raw)
            encoding = self.fc1(x)
        else:
            x = self.transformer(torch.cat((X, X_raw), dim=1))
            encoding = self.fc1(x)
        merged = torch.cat((encoding, X_raw, torch.ones((X_raw.shape[0], 1)).to(self.d)), dim=1)
        out = self.output(merged)
        return out
