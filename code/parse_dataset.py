import os

def parse_metafile(meta_file):
    """
    :param meta_file: string
    :return: celeb_ids: dictionary | format: {VoxCeleb1 ID: VGGFace1 ID, ...}
    """
    with open(meta_file, 'r') as f:
        lines = f.readlines()[1:]
    celeb_v2f = {}
    # celeb_f2v = {}
    for line in lines:
        # VoxCeleb1 ID, VGGFace1 ID, Gender, Nationality, Set
        ID, name, _, _, _ = line.rstrip().split('\t')
        celeb_v2f[ID] = name
        # celeb_f2v[name] = ID
    return celeb_v2f  # , celeb_f2v

def get_labels(voice_list, face_list):
    """ Take intersection between VoxCeleb1 and VGGFace1,
    and reorder pair with number starting from 0
    :param voice_list:
    :param face_list:
    :return: x_dict format:
    { (int) label_id : {'filepath': (str) filepath, 'name': (str) celeb_name, 'label_id': (int) label_id},
      ...}
    """
    voice_names = {item['name'] for item in voice_list}
    face_names = {item['name'] for item in face_list}
    names = voice_names & face_names  # s.intersection(t) ==> s & t

    voice_list = [item for item in voice_list if item['name'] in names]
    face_list = [item for item in face_list if item['name'] in names]

    names = sorted(list(names))
    label_dict = dict(zip(names, range(len(names))))
    voice_dict = {}
    face_dict = {}
    for item in voice_list:
        identity = label_dict[item['name']]
        item['label_id'] = identity
        voice_dict[identity] = item
    for item in face_list:
        identity = label_dict[item['name']]
        item['label_id'] = identity
        face_dict[identity] = item
    return voice_list, face_list, len(names), voice_dict, face_dict
    

def get_dataset_files(data_dir, data_ext, celeb_ids, split):
    # returned data_list format: [{'filepath': filepath, 'name': celeb_name}, {...}, ...]
    data_list = []
    # read data directory
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(data_ext):
                filepath = os.path.join(root, filename)
                # so hacky, be careful! 
                folder = filepath[len(data_dir):].split('/')[1]
                # TODO: what string format is celeb_name
                celeb_name = celeb_ids.get(folder, folder)
                if celeb_name.startswith(tuple(split)):
                    data_list.append({'filepath': filepath, 'name': celeb_name})
    return data_list

def get_dataset(data_params):
    celeb_ids_v2f = parse_metafile(data_params['meta_file'])  # , celeb_ids_f2v
    
    voice_list = get_dataset_files(data_params['voice_dir'],
                                   data_params['voice_ext'],
                                   celeb_ids_v2f,
                                   data_params['split'])
    face_list = get_dataset_files(data_params['face_dir'],
                                  data_params['face_ext'],
                                  celeb_ids_v2f,
                                  data_params['split'])
    return get_labels(voice_list, face_list)