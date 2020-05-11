import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, dim=768, n_filters = 100, filter_sizes = [2,3,4,5], n_labels = 4, 
                 dropout_prob = 0.2, length = 24):
        """
        :param dim: input vector dimension
        :param n_filters: number of filters to be applied
        :param filter_sizes: list of kernel sizes to be applied
        :param n_labels: number of output labels
        :param dropout_prob: dropout value
        :param length: input vector will be reshaped as matrix of wide * lenght, it must be > kernel_size and (dim % lenght) must be equal 0.
        """
        super(CNN, self).__init__()

        self.length = length
        self.dim = dim
        self.width = int(self.dim/self.length)
        self.n_filters = n_filters # supposed >> 4 * output_dim E.g. 100
        self.filter_sizes = filter_sizes
        self.output_dim = n_labels
        self.dropout_prob = dropout_prob
        if self.dim % self.length != 0 :
            print('ERROR: vector input dimension (dim = %d) must be divisible by selected lenght (%d)' %(self.dim,self.length))
            exit(-1)
        for f in filter_sizes:
            if f > self.length:
                print('ERROR: selected length (%d) must be bigger than any kernel size:' %(self.length), self.filter_sizes)
                exit(-2)
        #Possible more than one convolutional layer
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = self.width, #self.embedding_dim, #embedding_dim
                                              out_channels = self.n_filters, 
                                              kernel_size = fs)
                                    for fs in self.filter_sizes
                                    ])
        self.fc1 = nn.Linear(len(self.filter_sizes) * self.n_filters, self.n_filters)
        self.fc2 = nn.Linear(self.n_filters, int(self.n_filters/2))
        self.fc3 = nn.Linear(int(self.n_filters/2), int(self.n_filters/4))
        self.fc4 = nn.Linear(int(self.n_filters/4), self.output_dim)
        self.dropout = nn.Dropout(self.dropout_prob)

    def freeze_layers(self, layers):
        for param in layers.parameters():
            param.requires_grad = False   
        
    def forward(self, input):
        #input = (batch_size x cls)
        # Turn (batch_size x embedding_size) into (batch_size x width x lenth) for CNN
        view = input.view(input.size(0), self.width, self.length) 
        conved = [F.relu(conv(view)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim = 1)
        fc1_out = self.fc1(cat)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)
        output = self.fc4(fc3_out)
        return output, cat


class TEXT_ENSEMBLE(nn.Module):
    def __init__(self, bert, model_b):
        """
        :param bert: bert model that outputs its last hidden layer cls in the form (logits, cls)
        :param model_b: a model (like CNN with default parameters) that accepts cls outputs (batch_size x 768) 
        """
        super(TEXT_ENSEMBLE, self).__init__()
        self.bert = bert
        self.model_b = model_b
        self.config = bert.config
    
    def forward(self, input):
        _, cls = self.bert(input)
        return self.model_b(cls)