import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import DataLoader
from config import DATASET_PARAMETERS, NETWORKS_PARAMETERS
from parse_dataset import get_dataset
from network import get_network
from utils import Meter, cycle, save_model
from loss import *
from dataset import reload_batch_face, reload_batch_voice
from torchvision.utils import save_image

# dataset and dataloader
print('Parsing your dataset...')
voice_list, face_list, id_class_num, voice_dict, face_dict = get_dataset(DATASET_PARAMETERS)
NETWORKS_PARAMETERS['c']['output_channel'] = id_class_num

print('Preparing the datasets...')
voice_dataset = DATASET_PARAMETERS['voice_dataset'](voice_list,
                               DATASET_PARAMETERS['nframe_range'])
face_dataset = DATASET_PARAMETERS['face_dataset'](face_list)

print('Preparing the dataloaders...')
collate_fn = DATASET_PARAMETERS['collate_fn'](DATASET_PARAMETERS['nframe_range'])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size=1,
                          num_workers=DATASET_PARAMETERS['workers_num'],
                          collate_fn=collate_fn)
face_loader = DataLoader(face_dataset, shuffle=True, drop_last=True,
                         batch_size=DATASET_PARAMETERS['batch_size'],
                         num_workers=DATASET_PARAMETERS['workers_num'])

voice_iterator = iter(cycle(voice_loader))
face_iterator = iter(cycle(face_loader))

# model = torch.load('models/generator.pth')
e_net, e_optimizer = get_network('e', NETWORKS_PARAMETERS, train=False)  # voice embedding
g_net, g_optimizer = get_network('u', NETWORKS_PARAMETERS, train=False) 
g_net.eval()

for i in range(1):
  voiceB, voiceB_label = next(voice_iterator)
  faceA, faceA_label = next(face_iterator)  # real face

  faceB_items = [face_dict[v_label.item()] for v_label in voiceB_label]
  faceB = reload_batch_face(faceB_items)

  voiceB, faceA = voiceB.cuda(), faceA.cuda()
  # get voice embeddings
  embedding_B = e_net(voiceB)
  embedding_B = F.normalize(embedding_B).view(embedding_B.size()[0], -1)

  scaled_images = faceA * 2 - 1
  fake_faceB = g_net(scaled_images, embedding_B)
  fake_faceB = (fake_faceB + 1) / 2


  # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
  save_image(faceA, 'results/Input/img{}.png'.format(i+1000))
  save_image(fake_faceB, 'results/conversion/img{}.png'.format(i+1000))
  save_image(faceB, 'results/Obj/img{}.png'.format(i+1000))

