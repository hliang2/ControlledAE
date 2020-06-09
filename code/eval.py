import os
import time
import torch
# import torchaudio
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from network import get_network
from torch.utils.data import DataLoader
from config import DATASET_PARAMETERS, NETWORKS_PARAMETERS
from parse_dataset import get_dataset
from utils import Meter, cycle, save_model, test_image
from dataset import reload_batch_face, reload_batch_voice
from torchvision.utils import save_image
# from scipy.io.wavfile import write

sample_rate = 16000

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
                         batch_size=1,
                         num_workers=DATASET_PARAMETERS['workers_num'])

voice_iterator = iter(cycle(voice_loader))
face_iterator = iter(cycle(face_loader))

g_net_ours = torch.load('Gsave.pt', map_location=torch.device('cpu'))
g_net_yd, g_optimizer = get_network('g', NETWORKS_PARAMETERS, train=False)
e_net, e_optimizer = get_network('e', NETWORKS_PARAMETERS, train=False)  # voice embedding

# g_net.eval()

for i in range(1):
	test_image(i, e_net, g_net_ours, g_net_yd, 'all2', voice_loader, face_loader, face_dict)
# for i in range(10):
# 	# voiceB, voiceB_label = next(voice_iterator)
# 	faceA, faceA_label = next(face_iterator)  # real face
# 	# voiceB_label = voiceB_label.repeat(DATASET_PARAMETERS['batch_size'])
# 	# TODO: since voiceB and faceA in different identities,
# 	#  need to reuse load_voice and load_face to get corresponding faceB and voiceA
# 	# faceB_items = [face_dict[v_label.item()] for v_label in voiceB_label]
# 	voiceA_items = [voice_dict[f_label.item()] for f_label in faceA_label]
# 	# faceB = reload_batch_face(faceB_items)
# 	voiceA = reload_batch_voice(voiceA_items, DATASET_PARAMETERS['nframe_range'][1])

# 	# embedding_B = e_net(voiceB)
# 	# embedding_B = F.normalize(embedding_B).view(embedding_B.size()[0], -1)

# 	embedding_A = e_net(voiceA)
# 	embedding_A = F.normalize(embedding_A)

# 	# scaled_images = faceA * 2 - 1
# 	fake_faceA = g_net(embedding_A)
# 	# fake_faceB = (fake_faceB + 1) / 2
# 	save_image(fake_faceA, 'yandong_results/img{}.png'.format(i))
# 	save_image(faceA, 'yandong_results/inputimg{}.png'.format(i))