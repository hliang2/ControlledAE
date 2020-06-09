import string
import torchvision.transforms as transforms
from dataset import VoiceDataset, FaceDataset
from network import VoiceEmbedNet, Generator, UNet, FaceEmbedNet, Classifier
from utils import get_collate_fn


DATASET_PARAMETERS = {
    # TRAINING CONFIGURATIONS
    'num_batches': 400000,
    'print_stat_freq': 10000,

    # meta data provided by voxceleb1 dataset
    'meta_file': 'data/vox1_meta.csv',

    # voice dataset
    'voice_dir': 'data/fbank',
    'voice_ext': 'npy',

    # face dataset
    'face_dir': 'data/VGG_ALL_FRONTAL',
    'face_ext': '.jpg',

    # train data includes the identities
    # whose names start with the characters of 'FGH...XYZ' 
    'split': string.ascii_uppercase[5:],

    # dataloader
    'voice_dataset': VoiceDataset,
    'face_dataset': FaceDataset,
    'batch_size': 1,
    'nframe_range': [300, 800],
    'workers_num': 1,
    'collate_fn': get_collate_fn,

    # transform
    "transform": transforms.Compose(
        [transforms.Resize((64,64)),
         transforms.ToTensor(),
         ]),

    # test data
    'test_data': 'data/test_data/'
}


NETWORKS_PARAMETERS = {
    # VOICE EMBEDDING NETWORK (e)
    'e': {
        'network': VoiceEmbedNet,
        'input_channel': 64,
        'channels': [256, 384, 576, 864],
        'output_channel': 64, # the embedding dimension
        'model_path': 'pretrained_models/voice_embedding.pth',
    },
    # GENERATOR UNET (u)
    'u': {
        'network': UNet,
        'input_channel': 3,
        'channels': [],
        'output_channel': 3,
        'model_path': 'models/unet.pth',
    },
    # DISCRIMINATOR (d)
    'd': {
        'network': Classifier, # Discrminator is a special Classifier with 1 subject
        'input_channel': 64,
        'channels': [],
        'output_channel': 1,
        'model_path': 'models/discriminator.pth',
    },
    # CLASSIFIER (c)
    # TODO: change classifier to an actual net
    'c': {
        'network': Classifier,
        'input_channel': 64,
        'channels': [],
        'output_channel': -1,  # This parameter is depended on the dataset we used
        'model_path': 'models/classifier.pth',
    },
    # GENERATOR (g)
    # TODO: NO USE
    'g': {
        'network': Generator,
        'input_channel': 64,
        'channels': [1024, 512, 256, 128, 64],  # channels for deconvolutional layers
        'output_channel': 3,  # images with RGB channels
        'model_path': 'pretrained_models/generator.pth',
    },
    # FACE EMBEDDING NETWORK (f)
    # TODO: trainable to fixed
    'f': {
        'network': FaceEmbedNet,
        'input_channel': 3,
        'channels': [32, 64, 128, 256, 512],
        'output_channel': 64,
        'model_path': 'models/face_embedding.pth',
    },

    # OPTIMIZER PARAMETERS 
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,

    # MODE, use GPU or not
    'GPU': False,
}