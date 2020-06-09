import os
import torch
import shutil
import numpy as np
import torch.nn.functional as F

from network import get_network
from PIL import Image
from scipy.io import wavfile
from torch import topk
from torch.utils.data.dataloader import default_collate
from vad import read_wave, write_wave, frame_generator, vad_collector
from torchvision.utils import save_image
from dataset import reload_batch_face, reload_batch_voice
# from config import DATASET_PARAMETERS, NETWORKS_PARAMETERS

class Meter(object):
    # Computes and stores the average and current value
    def __init__(self, name, display, fmt=':f'):
        self.name = name
        self.display = display
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{' + self.display + self.fmt + '},'
        return fmtstr.format(**self.__dict__)

def get_collate_fn(nframe_range):
    def collate_fn(batch):
        min_nframe, max_nframe = nframe_range
        assert min_nframe <= max_nframe
        num_frame = np.random.randint(min_nframe, max_nframe+1)
        pt = np.random.randint(0, max_nframe-num_frame+1)
        batch = [(item[0][..., pt:pt+num_frame], item[1])
                 for item in batch]
        return default_collate(batch)
    return collate_fn

def cycle(dataloader):
    while True:
        for data, label in dataloader:
            yield data, label

def save_model(net, model_path):
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
       os.makedirs(model_dir)
    torch.save(net.state_dict(), model_path)

def retrieve_face(face_image, f_net, c_net, face_dict, k):
  f_net.eval()
  c_net.eval()
  result = c_net(f_net(face_image))
  res, ind = result.topk(k, largest=True)
  # print(ind)
  faces = [face_dict[i.item()] for i in ind.view(-1)]
  f_net.train()
  c_net.train()
  return faces

def get_cos(fake_image, real_image):
  return F.cosine_similarity(fake_image, real_image)

def test_image(iter_id, e_net, g_net_o, g_net_y, label, voice_loader, face_loader, face_dict):
    g_net_o.eval()
    g_net_y.eval()
    # f_net.eval()

    # g_net2 = torch.load('G.pt')
    # c_net = torch.load(c_net, 'C.pt')
    # f_net = torch.load(f_net, 'face.pt')
    # g_net2 = torch.load('G_addedA.pt')
    if not os.path.exists('new_results/{}'.format(label)):
      os.makedirs('new_results/{}'.format(label))
    if not os.path.exists('new_results/{}/{}/Input'.format(label, iter_id)):
      os.makedirs('new_results/{}/{}/Input'.format(label, iter_id))
      os.makedirs('new_results/{}/{}/Obj'.format(label, iter_id))
      os.makedirs('new_results/{}/{}/conversion'.format(label, iter_id))
      os.makedirs('new_results/{}/{}/conversion_y'.format(label, iter_id))
      # os.makedirs('results/{}/{}/conversion2'.format(label, iter_id))
      # os.makedirs('results/{}/{}/retrieve'.format(label, iter_id))

    voice_iterator = iter(cycle(voice_loader))
    face_iterator = iter(cycle(face_loader))
    # voiceB, voiceB_label = next(voice_iterator)
    #   # faceA, faceA_label = next(face_iterator)  # real face

    # faceB_items = [face_dict[v_label.item()] for v_label in voiceB_label]
    # faceB = reload_batch_face(faceB_items)

    # voiceB, faceA = voiceB.cuda(), faceA.cuda()
    # get voice embeddings
    # embedding_B = e_net(voiceB)
    # embedding_B_y = F.normalize(embedding_B)
    # embedding_B = F.normalize(embedding_B).view(embedding_B.size()[0], -1)
    # cos1, cos2, cos3, cos4, cos5 = 0, 0, 0, 0, 0
    for i in range(200):
      print('********iter{}*********'.format(i))
      faceA, faceA_label = next(face_iterator)  # real face
      voiceB, voiceB_label = next(voice_iterator)

      faceB_items = [face_dict[v_label.item()] for v_label in voiceB_label]
      faceB = reload_batch_face(faceB_items)

      # voiceB, faceA = voiceB.cuda(), faceA.cuda()
      # get voice embeddings
      embedding_B = e_net(voiceB)
      embedding_B_y = F.normalize(embedding_B)
      embedding_B = F.normalize(embedding_B).view(embedding_B.size()[0], -1)

      
      # face_iterator = iter(cycle(face_loader))
      # faceA, faceA_label = next(face_iterator)  # real face
      scaled_images = faceA * 2 - 1
      fake_faceB = g_net_o(scaled_images, embedding_B)
      fake_faceB = (fake_faceB + 1) / 2

      fake_faceB2 = g_net_y(embedding_B_y)
      # fake_faceB2 = (fake_faceB2 + 1) / 2

      # print(f_net(fake_faceB))

      # cos1 += get_cos(f_net(fake_faceB.cuda()), f_net(faceB.cuda()))
      # cos2 += get_cos(f_net(fake_faceB2.cuda()), f_net(faceB.cuda()))

      # cos3 += get_cos(f_net(fake_faceB.cuda()), f_net(faceA.cuda()))
      # cos4 += get_cos(f_net(fake_faceB2.cuda()), f_net(faceA.cuda()))

      # cos5 += get_cos(f_net(faceB.cuda()), f_net(faceA.cuda()))

      # retrieve = retrieve_face(fake_faceB, f_net, c_net, face_dict, 5)
      # if not os.path.exists('results/{}/{}/retrieve/{}'.format(label, iter_id, i+1)):
      #   os.makedirs('results/{}/{}/retrieve/{}'.format(label, iter_id, i+1))
      # for j in range(len(retrieve)):
      #   # print(type(retrieve[j]))
      #   face = reload_batch_face([retrieve[j]])
      #   save_image(face, 'results/{}/{}/retrieve/{}/{}.png'.format(label, iter_id, i+1, j+1))

      save_image(faceA, 'new_results/{}/{}/Input/img{}.png'.format(label, iter_id, i+1))
      save_image(fake_faceB, 'new_results/{}/{}/conversion/img{}.png'.format(label, iter_id, i+1))
      save_image(fake_faceB2, 'new_results/{}/{}/conversion_y/img{}.png'.format(label, iter_id, i+1))
      save_image(faceB, 'new_results/{}/{}/Obj/img{}.png'.format(label, iter_id, i+1))
    # g_net.train()
    # f_net.train()

    # print(cos1.item()/49, cos2.item()/49, cos3.item()/49, cos4.item()/49, cos5.item()/49)
    # return cos1.item()/49

def rm_sil(voice_file, vad_obj):
    """
       This code snippet is basically taken from the repository
           'https://github.com/wiseman/py-webrtcvad'
       It removes the silence clips in a speech recording
    """
    audio, sample_rate = read_wave(voice_file)
    frames = frame_generator(20, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 20, 50, vad_obj, frames)

    if os.path.exists('tmp/'):
       shutil.rmtree('tmp/')
    os.makedirs('tmp/')

    wave_data = []
    for i, segment in enumerate(segments):
        segment_file = 'tmp/' + str(i) + '.wav'
        write_wave(segment_file, segment, sample_rate)
        wave_data.append(wavfile.read(segment_file)[1])
    shutil.rmtree('tmp/')

    if wave_data:
       vad_voice = np.concatenate(wave_data).astype('int16')
    return vad_voice

def get_fbank(voice, mfc_obj):
    # Extract log mel-spectrogra
    fbank = mfc_obj.sig2logspec(voice).astype('float32')

    # Mean and variance normalization of each mel-frequency 
    fbank = fbank - fbank.mean(axis=0)
    fbank = fbank / (fbank.std(axis=0)+np.finfo(np.float32).eps)

    # If the duration of a voice recording is less than 10 seconds (1000 frames),
    # repeat the recording until it is longer than 10 seconds and crop.
    full_frame_number = 1000
    init_frame_number = fbank.shape[0]
    while fbank.shape[0] < full_frame_number:
          fbank = np.append(fbank, fbank[0:init_frame_number], axis=0)
          fbank = fbank[0:full_frame_number,:]
    return fbank


def voice2face(e_net, g_net, voice_file, vad_obj, mfc_obj, GPU=True):
    vad_voice = rm_sil(voice_file, vad_obj)
    fbank = get_fbank(vad_voice, mfc_obj)
    fbank = fbank.T[np.newaxis, ...]
    fbank = torch.from_numpy(fbank.astype('float32'))
    
    if GPU:
        fbank = fbank.cuda()
    embedding = e_net(fbank)
    embedding = F.normalize(embedding)
    face = g_net(embedding)
    return face

def experiment(e_net, voice_iterator, face_iterator, face_dict, voice_dict):
    g_net = torch.load('G.pt')
    g_net.eval()
    for i in range(200):
        voiceB, voiceB_label = next(voice_iterator)
        faceA, faceA_label = next(face_iterator)  # real face
        voiceB_label = voiceB_label.repeat(1)
        # TODO: since voiceB and faceA in different identities,
        #  need to reuse load_voice and load_face to get corresponding faceB and voiceA
        faceB_items = [face_dict[v_label.item()] for v_label in voiceB_label]
        voiceA_items = [voice_dict[f_label.item()] for f_label in faceA_label]

        path = voice_dict[voiceB_label.item()]['filepath']
        path = path.split('/')[3]
        name = voice_dict[voiceB_label.item()]['name']
        faceB = reload_batch_face(faceB_items)
        # voiceA = reload_batch_voice(voiceA_items, DATASET_PARAMETERS['nframe_range'][1])

        voiceB, voiceB_label = voiceB.cuda(), voiceB_label.cuda()
        faceA, faceA_label = faceA.cuda(), faceA_label.cuda()
        faceB = faceB.cuda()

        embedding_B = e_net(voiceB)
        embedding_B = F.normalize(embedding_B).view(embedding_B.size()[0], -1)

        scaled_images = faceB * 2 - 1
        fake_faceB = g_net(scaled_images, embedding_B)
        fake_faceB = (fake_faceB + 1) / 2

        if not os.path.exists('results/same2/{}'.format(i+1)):
            os.makedirs('results/same2/{}'.format(i+1))

        save_image(faceB, 'results/same2/{}/{}_{}.png'.format(i+1, path, name))
        save_image(fake_faceB, 'results/same2/{}/fake{}.png'.format(i+1, name))