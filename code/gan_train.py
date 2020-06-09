import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import DataLoader
from config import DATASET_PARAMETERS, NETWORKS_PARAMETERS
from parse_dataset import get_dataset
from network import get_network
from utils import Meter, cycle, save_model, test_image, experiment
from loss import *
from dataset import reload_batch_face, reload_batch_voice

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

# networks, Fe, Fg, Fd (f+d), Fc (f+c)
print('Initializing networks...')
e_net, e_optimizer = get_network('e', NETWORKS_PARAMETERS, train=False)  # voice embedding
# g_net, g_optimizer = get_network('g', NETWORKS_PARAMETERS, train=True)
f_net, f_optimizer = get_network('f', NETWORKS_PARAMETERS, train=True)
g_net, g_optimizer = get_network('u', NETWORKS_PARAMETERS, train=True)  # unet
d_net, d_optimizer = get_network('d', NETWORKS_PARAMETERS, train=True)  # discriminator
c_net, c_optimizer = get_network('c', NETWORKS_PARAMETERS, train=True)  # classifier, train=False

d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=1, gamma=0.96)
g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=1, gamma=0.96)

experiment(e_net, voice_iterator, face_iterator, face_dict, voice_dict)
# Meters for recording the training status
iteration = Meter('Iter', 'sum', ':5d')
# data_time = Meter('Data', 'sum', ':4.2f')
# batch_time = Meter('Time', 'sum', ':4.2f')

meter_D_real = Meter('D_real', 'avg', ':3.2f')
meter_D_fake = Meter('D_fake', 'avg', ':3.2f')
meter_D = Meter('D', 'avg', ':3.2f')
meter_C_real = Meter('C_real', 'avg', ':3.2f')
meter_GD_fake = Meter('G_D_fake', 'avg', ':3.2f')
meter_GC_fake = Meter('G_C_fake', 'avg', ':3.2f')
meter_G_L2_fake = Meter('G_l2_fake', 'avg', ':3.2f')
meter_G = Meter('G', 'avg', ':3.2f')
""" """

g_net = torch.load('G_addedA.pt')
g_net.train()
# d_net = torch.load('FD.pt')
# d_net.train()

print('Training models...')
min_g_loss = None
min_d_loss = None
min_c_loss = None
for it in range(DATASET_PARAMETERS['num_batches']):
    # data
    start_time = time.time()

    voiceB, voiceB_label = next(voice_iterator)
    faceA, faceA_label = next(face_iterator)  # real face
    voiceB_label = voiceB_label.repeat(DATASET_PARAMETERS['batch_size'])
    # TODO: since voiceB and faceA in different identities,
    #  need to reuse load_voice and load_face to get corresponding faceB and voiceA
    faceB_items = [face_dict[v_label.item()] for v_label in voiceB_label]
    voiceA_items = [voice_dict[f_label.item()] for f_label in faceA_label]
    faceB = reload_batch_face(faceB_items)
    voiceA = reload_batch_voice(voiceA_items, DATASET_PARAMETERS['nframe_range'][1])
    # noise = 0.05 * torch.randn(DATASET_PARAMETERS['batch_size'], 64, 1, 1)  # shape 4d!

    # print(voiceB.shape)
    # torch.Size([64, 64, 438])
    # print(faceA.shape)
    # torch.Size([64, 3, 64, 64])
    # use GPU or not
    if NETWORKS_PARAMETERS['GPU']:
        voiceB, voiceB_label = voiceB.cuda(), voiceB_label.cuda()
        faceA, faceA_label = faceA.cuda(), faceA_label.cuda()
        faceB, voiceA = faceB.cuda(), voiceA.cuda()
        # real_label, fake_label = real_label.cuda(), fake_label.cuda()
        # noise = noise.cuda()
    # data_time.update(time.time() - start_time)

    # TODO: scale the input images, notice when inference ??
    # scaled_images = face * 2 - 1

    # get voice embeddings
    embedding_B = e_net(voiceB)
    embedding_B = F.normalize(embedding_B).view(embedding_B.size()[0], -1)
    # print(embedding_B.shape) #(64, 64)
    # introduce some permutations to voice --> deprecated
    # embeddings = embeddings + noise
    # embeddings = F.normalize(embeddings)


    # TODO: introduce some permutations to image !!!

    # ============================================
    #            TRAIN THE DISCRIMINATOR
    # ============================================

    # if it != 1 and it % 10 == 1:
    f_optimizer.zero_grad()
    d_optimizer.zero_grad()
    c_optimizer.zero_grad()

    # 0. get generated faces
    scaled_images = faceA * 2 - 1
    fake_faceB = g_net(scaled_images, embedding_B)
    fake_faceB = (fake_faceB + 1) / 2
    # 1. Train with real images
    D_real_A = d_net(f_net(faceA))
    D_real_A_loss = true_D_loss(torch.sigmoid(D_real_A))

    # 2. Train with fake images
    D_fake_B = d_net(f_net(fake_faceB).detach())
    # D_fake = d_net(f_net(fake_face.detach()))  # TODO: is detach necessary here ???
    D_fake_B_loss = fake_D_loss(torch.sigmoid(D_fake_B))

    # 3. Train with identity / gender classification
    real_classification = c_net(f_net(faceA))
    C_real_loss = identity_D_loss(real_classification, faceA_label)

    # D_real_loss = F.binary_cross_entropy(torch.sigmoid(D_real), real_label)
    # D_fake_loss = F.binary_cross_entropy(torch.sigmoid(D_fake), fake_label)

    # backprop
    D_loss = D_real_A_loss + D_fake_B_loss + C_real_loss
    # update meters
    meter_D_real.update(D_real_A_loss.item())
    meter_D_fake.update(D_fake_B_loss.item())
    meter_C_real.update(C_real_loss.item())
    meter_D.update(D_loss.item())

    D_loss.backward()

    f_optimizer.step()
    c_optimizer.step()
    d_optimizer.step()

    # =========================================
    #            TRAIN THE GENERATOR
    # =========================================
    g_optimizer.zero_grad()

    # 0. get generated faces
    fake_faceB = g_net(scaled_images, embedding_B)
    fake_faceB = (fake_faceB + 1) / 2

    # 0.5 Train with L2 loss with A
    l2lossA = l1_loss_G(fake_faceB, faceA, 1)

    # 1. Train with discriminator
    D_fake_B = d_net(f_net(fake_faceB))
    D_B_loss = true_D_loss(torch.sigmoid(D_fake_B))

    # 2. Train with classifier
    fake_classfication = c_net(f_net(fake_faceB))
    C_fake_loss = identity_D_loss((fake_classfication), voiceB_label)
    # C_fake_loss = F.nll_loss(F.log_softmax(fake_classfication, 1), voice_label)

    # GD_fake_loss = F.binary_cross_entropy(torch.sigmoid(D_fake), real_label)
    # GC_fake_loss = F.nll_loss(F.log_softmax(fake_classfication, 1), voice_label)

    # 3. Train with L2 loss
    l2lossB = l1_loss_G(fake_faceB, faceB)

    # 4. Train with consistency loss
    # TODO: to be tested, after getting embedding_A and ??
    # scaled_fake = fake_faceB * 2 - 1
    # # get voice embeddings
    # embedding_A = e_net(voiceA)
    # embedding_A = F.normalize(embedding_A).view(embedding_A.size()[0], -1)
    # fake_faceA = g_net(fake_faceB, embedding_A)
    # fake_faceA = (fake_faceA + 1) / 2
    # consistency_loss = l1_loss_G(fake_faceA, faceA)

    # backprop
    # G_loss = l2lossA + l2lossB
    G_loss = C_fake_loss + l2lossA + l2lossB + D_B_loss
    G_loss.backward()
    meter_GD_fake.update(D_B_loss.item())
    # meter_GC_fake.update(C_fake_loss.item())
    # meter_G_L2_fake.update(l2loss.item() + consistency_loss.item())
    meter_G_L2_fake.update(l2lossB.item())
    meter_G.update(G_loss.item())
    g_optimizer.step()

    # batch_time.update(time.time() - start_time)

    # print status
    if it % DATASET_PARAMETERS['print_stat_freq'] == 0:
        f_net = torch.load('face.pt')
        c_net = torch.load('C.pt')
        
        cos = test_image(it, f_net, c_net, e_net, g_net, voice_loader, face_loader, 'double_test', face_dict)
        print(iteration, meter_D, meter_G, meter_GD_fake, meter_G_L2_fake, 'cos:', cos)
        # data_time.reset()
        # batch_time.reset()
        meter_G.reset()
        meter_G_L2_fake.reset()
        meter_D.reset()
        # meter_D_real.reset()
        # meter_D_fake.reset()
        # meter_C_real.reset()
        meter_GD_fake.reset()
        # meter_GC_fake.reset()

        # snapshot
        # save_model(g_net, NETWORKS_PARAMETERS['u']['model_path'])

        # cos = test_image(it, f_net, c_net, e_net, g_net, voice_loader, face_loader, 'final_test', face_dict)

    # save model for debugging purpose
    # if min_g_loss is None or G_loss < min_g_loss:
    #     min_g_loss = G_loss
    #     torch.save(g_net, 'G.pt')
    # if min_d_loss is None or D_loss < min_d_loss:
    #     min_d_loss = D_loss
    #     torch.save(d_net, 'FD.pt')
    # if min_c_loss is None or C_real_loss < min_c_loss:
    #     min_d_loss = D_loss
    #     torch.save(c_net, 'C.pt')
    #     torch.save(f_net, 'face.pt')

    iteration.update(1)