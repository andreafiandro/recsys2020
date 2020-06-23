import torch
import torch.nn as nn

class Multi_Label_RCE_Loss(nn.Module): # higher RCE values is a good thing
    def __init__(self, ctr):
        """
        :param ctr: Tensor of positive rate for each val set label, computed as Positives/tot, tensor of n_labels dimension
        """
        super(Multi_Label_RCE_Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none') #need of loss per label so no reduction
        self.ctr = ctr

    def forward(self, predictions, targets): #input dim = targets dim =  (N, n_labels) where N = Batch_size
        """
        :param predictions: tensor of batch predictions logits
        :param targets: tensor of batch targets
        """
        cross_entropies = self.criterion(predictions, targets) #(Batch_size, n_labels)
        cross_entropies = cross_entropies.mean(0) #batch loss mean per label
        data_ctr = targets.clone().detach() #torch.zeros(targets.size()).cuda()
        for i in range(len(targets)):
            data_ctr[i] = self.ctr
        strawman_cross_entropies = self.criterion(data_ctr, targets).mean(0)
        return (cross_entropies / strawman_cross_entropies).mul_(-1).add_(1.0).mul_(100.0)
