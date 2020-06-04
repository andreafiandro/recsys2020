import torch.nn as nn


class Multi_Label_RCE_Loss(nn.Module):  # La rce più grande è meglio è!
    def __init__(self, ctr):
        """
        :param ctr: Tensore dei ratei dei positivi per etichetta del val set,
        calcolati come Positives/tot, Tensore lungo n_labels
        """
        super(Multi_Label_RCE_Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')  # need of
        # loss per label so no reduction
        self.ctr = ctr

    def forward(self, predictions, targets):  # input dim = targets dim =
        # (N, n_labels) where N = Batch_size
        """
        :param predictions: tensor of batch predictions logits
        :param targets: tensor of batch targets
        """
        cross_entropies = self.criterion(predictions, targets)
        # (Batch_size, n_labels)
        cross_entropies = cross_entropies.mean(0)
        # batch loss mean per label
        data_ctr = targets.clone().detach()
        # torch.zeros(targets.size()).cuda()
        for i in range(len(targets)):
            data_ctr[i] = self.ctr
        strawman_cross_entropies = self.criterion(data_ctr, targets).mean(0)
        return (cross_entropies /
                strawman_cross_entropies).mul_(-1).add_(1.0).mul_(100.0)
