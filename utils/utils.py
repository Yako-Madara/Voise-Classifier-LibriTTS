import pandas as pd
import torch

def get_labels():
    """Создает и сохраняет словарь вида {speaker_id: gender}
    gender: 1-M, 0-F
    """
    speakers = pd.read_csv('./data/LibriTTS/speakers.tsv',sep='\t', header=0, index_col=False)
    speakers_dict = dict()
    for id_, gender in zip(speakers["READER"], speakers["GENDER"]):
        if gender =='F':
            g = 0
        elif gender == 'M':
            g = 1
        speakers_dict[id_] = g
    torch.save(speakers_dict, './data/labels/gender_labels')

