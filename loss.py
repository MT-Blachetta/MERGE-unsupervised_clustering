#import utils
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
#EPS = 1e-6

class EntLoss(nn.Module):
    def __init__(self, args, lam1, lam2, pqueue=None):
        super(EntLoss, self).__init__()
        self.lam1 = lam1
        self.lam2 = lam2
        self.pqueue = pqueue
        self.args = args
    
    def forward(self, feat1, feat2):
        probs1 = torch.nn.functional.softmax(feat1, dim=-1)
        probs2 = torch.nn.functional.softmax(feat2, dim=-1)
        loss = dict()
        loss['kl'] = 0.5 * (KL(probs1, probs2, self.args) + KL(probs2, probs1, self.args))

        sharpened_probs1 = torch.nn.functional.softmax(feat1/self.args['tau'], dim=-1)
        sharpened_probs2 = torch.nn.functional.softmax(feat2/self.args['tau'], dim=-1)
        loss['eh'] = 0.5 * (EH(sharpened_probs1, self.args) + EH(sharpened_probs2, self.args))

        # whether use historical data
        loss['he'] = 0.5 * (HE(sharpened_probs1, self.args) + HE(sharpened_probs2, self.args))

        loss['final'] = loss['kl'] + ((1+self.lam1)*loss['eh'] - self.lam2*loss['he'])
        return loss['final']

def KL(probs1, probs2, args):
    kl = (probs1 * (probs1 + args['EPS']).log() - probs1 * (probs2 + args['EPS']).log()).sum(dim=1)
    kl = kl.mean()
    #torch.distributed.all_reduce(kl)
    return kl

def CE(probs1, probs2, args):
    ce = - (probs1 * (probs2 + args['EPS']).log()).sum(dim=1)
    ce = ce.mean()
    #torch.distributed.all_reduce(ce)
    return ce

def HE(probs, args): 
    mean = probs.mean(dim=0)
    #torch.distributed.all_reduce(mean)
    ent  = - (mean * (mean + args['EPS']).log()).sum()
    return ent

def EH(probs, args):
    ent = - (probs * (probs + args['EPS']).log()).sum(dim=1)
    mean = ent.mean()
    #torch.distributed.all_reduce(mean)
    return mean


class SCAN_consistencyLoss(nn.Module):
    def __init__(self):
        super(SCAN_consistencyLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        #self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        #entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        #total_loss = consistency_loss# - self.entropy_weight * entropy_loss
        
        return consistency_loss

class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
        return loss


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 0.00001

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))
