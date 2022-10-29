import argparse
import pickle

import numpy as np
import torch
import torchaudio
import torchvision
from catboost import CatBoostClassifier
from torch import Tensor
from torch.nn import Module

from models.resnet import BasicBlock, ResNet, ResNet18

SAMPLE_RATE = 24000
N_MELS = 128
N_FFT = 1024

gender = {0: "FEMALE", 1: "MALE"}

sample_transforms = torchvision.transforms.Compose([
    torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                        n_fft=N_FFT, 
                                        n_mels=N_MELS,
                                        normalized=True),
    torchvision.transforms.Resize((128, 128))
])

model = torch.load("./parameters/model.pkl")


def get_audiofile_tensor(path: str, transforms = sample_transforms) -> Tensor:
    """Функция, преобразующая аудиофайл в двумерный тензор

    Args:
        path (str): путь до аудио файла

    Returns:
        Tensor: _description_
    """
    waveform, sr =torchaudio.load(path)
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)(waveform)
    waveform_tensor = transforms(waveform)
    return torch.unsqueeze(waveform_tensor, 0)

def get_numpy_vector(X: Tensor, model: Module, device: str) -> np.ndarray:
    """Функция извлечения признаков из двумерного тензора в одномерный numpy вектор

    Args:
        X (Tensor): 
        model (Module): обученная нейронная сеть
        device (str): устройство инференса

    Returns:
        np.ndarray: 
    """

    X = X.to(device)
    out = model(X, True)
    return out.cpu().detach().numpy()

def get_pca(path: str = './parameters/pca.pkl'):
    """Функция загрузки обученого объекта класса sklearn.decomposition.PCA

    Args:
        path (str, optional): путь до сохраненного обученного объекта PCA. Defaults to './parameters/pca.pkl'.

    Returns:
        :обученный объект класса sklearn.decomposition.PCA.
    """

    with open(path, 'rb') as pickle_file:
        pca_reload = pickle.load(pickle_file)
    return pca_reload

def get_classifier(path: str = './parameters/clf_model.pkl'):
    """

    Args:
        path (str, optional): путь до сохраненного классификатора CatBoostClassifier. 
        Defaults to './parameters/clf_model.pkl'.

    Returns:
        : CatBoostClassifier
    """

    clf = CatBoostClassifier()
    clf.load_model(path)
    return clf

def device_definition() -> str:
    """Определение устройства, на котором будет исполняться нейронная сеть.

    Returns:
        str: 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Обязательный аргумент. Путь до аудиофайла с голосом, который необходимо классифицировать.",
    )

    namespace_arg = parser.parse_args()
    X = get_audiofile_tensor(namespace_arg.path)
    device = device_definition()
    model = model.to(device)
    model.eval()
    X_numpy = get_numpy_vector(X, model, device)
    pca = get_pca()
    X_pca = pca.transform(X_numpy)
    clf = get_classifier()
    prediction = clf.predict(X_pca)
    print(gender[int(prediction[0])])
    