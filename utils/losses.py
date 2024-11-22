import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, true_labels=None, T=1, alpha=0.5):

    assert student_logits.dim() == 2 and teacher_logits.dim() == 2, 
    l, v = student_logits.size()
    

    soft_targets = F.softmax(teacher_logits / T, dim=-1)

    log_soft_student = F.log_softmax(student_logits / T, dim=-1)
    

    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    

    loss_kd = kl_div_loss(log_soft_student, soft_targets) * (T ** 2)
    
    if true_labels is not None:

        assert true_labels.dim() == 1 and true_labels.size(0) == l, 
        

        loss_ce = F.cross_entropy(student_logits, true_labels)
        
        loss = alpha * loss_kd + (1 - alpha) * loss_ce
    else:
        loss = loss_kd
    
    return loss



def distillation_loss_bias(student_logits, teacher_logits, T=1, train=False):

    assert student_logits.dim() == 2 and teacher_logits.dim() == 2, 
    l, v = student_logits.size()
    

    soft_targets = F.softmax(teacher_logits / T, dim=-1)

    log_soft_student = F.log_softmax(student_logits / T, dim=-1)
    
    if train:
        kl_div_loss = nn.KLDivLoss(reduction='none')
    else:
        kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    
    loss_kd = kl_div_loss(log_soft_student, soft_targets) * (T ** 2)

    if train:
        loss_kd = loss_kd.sum(dim=-1)
    
    return loss_kd
############################################################################################################################################




