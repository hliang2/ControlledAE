import torch
import torch.nn as nn
from config import NETWORKS_PARAMETERS

# ============ Loss ===============
# move the data to cuda if necessary
def l2_loss_G(out, input, weight=10.0):
    criterion = nn.MSELoss()
    return weight * criterion(out, input)


def l1_loss_G(out, input, weight=10.0):
    criterion = nn.L1Loss()
    return weight * criterion(out, input)


def true_D_loss(out):
    # labels need to be in cuda, size: (batch_size, 1)
    # scale by 0.9!
    batch_size = out.size(0)
    labels = 0.9 * torch.ones((batch_size, 1))
    if NETWORKS_PARAMETERS['GPU']:
        labels = labels.cuda()
    criterion = nn.BCELoss()
    return criterion(out, labels)


def fake_D_loss(out):
    batch_size = out.size(0)
    labels = torch.zeros((batch_size, 1))
    if NETWORKS_PARAMETERS['GPU']:
        labels = labels.cuda()
    criterion = nn.BCELoss()
    return criterion(out, labels)


def gender_D_loss(out, label):
    """ Loss for gender classifier discriminator
    :param out:
    :param label: scalar value
    :return:
    """
    batch_size = out.size(0)
    if label == 0:
        labels = torch.zeros(batch_size)
    else:  # if label == 1
        labels = torch.ones(batch_size)
    if NETWORKS_PARAMETERS['GPU']:
        labels = labels.cuda()
    criterion = nn.CrossEntropyLoss()
    return criterion(out, labels.long())


def identity_D_loss(out, labels):
    """ Loss for identity classifier discriminator
    """
    if NETWORKS_PARAMETERS['GPU']:
        labels = labels.cuda()
    criterion = nn.CrossEntropyLoss()
    return criterion(out, labels.long())