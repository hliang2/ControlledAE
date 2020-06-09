import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def load_voice(voice_item):
    voice_data = np.load(voice_item['filepath'])
    voice_data = voice_data.T.astype('float32')
    voice_label = voice_item['label_id']
    return voice_data, voice_label

def load_face(face_item, trans):
    face_data = Image.open(face_item['filepath']).convert('RGB').resize([64, 64])
    if trans is None:
        trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    face_data = trans(face_data)

    # face_data = np.transpose(np.array(face_data), (2, 0, 1))
    # face_data = ((face_data - 127.5) / 127.5).astype('float32')
    face_label = face_item['label_id']
    return face_data, face_label

def get_randomized_voice(voice_item, crop_nframe):
    voice_data, voice_label = load_voice(voice_item)
    assert crop_nframe <= voice_data.shape[1]
    pt = np.random.randint(voice_data.shape[1] - crop_nframe + 1)
    voice_data = voice_data[:, pt:pt + crop_nframe]
    return voice_data, voice_label

def get_randomized_face(face_item, trans=None):
    face_data, face_label = load_face(face_item, trans)
    """
    if np.random.random() > 0.5:
        face_data = np.flip(face_data, axis=2).copy()
    """
    return face_data, face_label

def reload_batch_voice(voice_items, crop_nframe):
    tmp_list = [torch.from_numpy(get_randomized_voice(item, crop_nframe)[0]).unsqueeze(0) for item in voice_items]
    return torch.cat(tmp_list, dim=0)

def reload_batch_face(face_items, trans=None):
    tmp_list = [get_randomized_face(item, trans)[0].unsqueeze(0) for item in face_items]
    return torch.cat(tmp_list, dim=0)


class VoiceDataset(Dataset):
    def __init__(self, voice_list, nframe_range):
        self.voice_list = voice_list
        self.crop_nframe = nframe_range[1]

    def __getitem__(self, index):
        voice_item = self.voice_list[index]
        return get_randomized_voice(voice_item, self.crop_nframe)

    def __len__(self):
        return len(self.voice_list)

class FaceDataset(Dataset):
    def __init__(self, face_list, transform=None):
        self.face_list = face_list
        self.transform = transform

    def __getitem__(self, index):
        face_item = self.face_list[index]
        return get_randomized_face(face_item, self.transform)

    def __len__(self):
        return len(self.face_list)
