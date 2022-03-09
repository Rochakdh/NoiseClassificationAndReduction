# FILE_NAME = ""
# def get_uploaded_file(file_name):
#     FILE_NAME = file_name

def implemeter(FILE_NAME):
    import librosa
    import pandas as pd
    import os
    import datetime
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import IPython.display as ipd
    import librosa.display
    import scipy
    import glob
    import numpy as np
    import math
    import warnings
    import pickle
    from sklearn.utils import shuffle
    import zipfile
    from keras.models import load_model
    import soundfile as sf
    import paths_src

    windowLength = 256
    overlap = round(0.25 * windowLength)  # overlap of 75%
    ffTLength = windowLength
    inputFs = 48e3
    fs = 16e3

    #locate sekte model for denoise

    #new_model = load_model("C:\\Users\\WINDROID\\Desktop\\model\\sekte_model.h5") 
    new_model = load_model(paths_src.SKETE_MODEL_PATH)   #new

    numFeatures = ffTLength//2 + 1
    numSegments = 8
    print("windowLength:", windowLength)
    print("overlap:", overlap)
    print("ffTLength:", ffTLength)
    print("inputFs:", inputFs)
    print("fs:", fs)
    print("numFeatures:", numFeatures)
    print("numSegments:", numSegments)


    sr = 16000
    # try:
    #     audioTest, sr = librosa.load(f"C:\\Users\\WINDROID\\Desktop\\{FILE_NAME}", sr=sr)
    # except:
    #     print(FILE_NAME)
    #     audioTest, sr = librosa.load(os.path(FILE_NAME), sr=sr)
    print("---------------------------------------------------------------------------")
    print(FILE_NAME)
    #for record
    #recorded audio files get stored to this path
    if(FILE_NAME == paths_src.RECORDED_FOLDER_PATH):
        audioTest, sr = librosa.load(f"{FILE_NAME}/{paths_src.RECORDED_FILE_NAME}", sr=sr)
    #for browse
    else:
        audioTest, sr = librosa.load(f"{paths_src.BROWSE_PATH}/{FILE_NAME}", sr=sr)
        
    noisyAudio = audioTest
    print(noisyAudio)
    ipd.Audio(data=noisyAudio, rate=fs)  # load a local WAV file


    def prepare_input_features(stft_features):
        # Phase Aware Scaling: To avoid extreme differences (more than
        # 45 degree) between the noisy and clean phase, the clean spectral magnitude was encoded as similar to [21]:
        noisySTFT = np.concatenate(
            [stft_features[:, 0:numSegments-1], stft_features], axis=1)
        stftSegments = np.zeros(
            (numFeatures, numSegments, noisySTFT.shape[1] - numSegments + 1))

        for index in range(noisySTFT.shape[1] - numSegments + 1):
            stftSegments[:, :, index] = noisySTFT[:, index:index + numSegments]
        return stftSegments


    class FeatureExtractor:
        def __init__(self, audio, *, windowLength, overlap, sample_rate):
            self.audio = audio
            self.ffT_length = windowLength
            self.window_length = windowLength
            self.overlap = overlap
            self.sample_rate = sample_rate
            self.window = scipy.signal.hamming(self.window_length, sym=False)

        def get_stft_spectrogram(self):
            return librosa.stft(self.audio, n_fft=self.ffT_length, win_length=self.window_length, hop_length=self.overlap,
                                window=self.window, center=True)

        def get_audio_from_stft_spectrogram(self, stft_features):
            return librosa.istft(stft_features, win_length=self.window_length, hop_length=self.overlap,
                                window=self.window, center=True)

        def get_mel_spectrogram(self):
            return librosa.feature.melspectrogram(self.audio, sr=self.sample_rate, power=2.0, pad_mode='reflect',
                                                n_fft=self.ffT_length, hop_length=self.overlap, center=True)

        def get_audio_from_mel_spectrogram(self, M):
            return librosa.feature.inverse.mel_to_audio(M, sr=self.sample_rate, n_fft=self.ffT_length, hop_length=self.overlap,
                                                        win_length=self.window_length, window=self.window,
                                                        center=True, pad_mode='reflect', power=2.0, n_iter=32, length=None)


    noiseAudioFeatureExtractor = FeatureExtractor(
        noisyAudio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
    noise_stft_features = noiseAudioFeatureExtractor.get_stft_spectrogram()

    # Paper: Besides, spectral phase was not used in the training phase.
    # At reconstruction, noisy spectral phase was used instead to
    # perform in- verse STFT and recover human speech.
    noisyPhase = np.angle(noise_stft_features)
    print(noisyPhase.shape)
    noise_stft_features = np.abs(noise_stft_features)

    mean = np.mean(noise_stft_features)
    std = np.std(noise_stft_features)
    noise_stft_features = (noise_stft_features - mean) / std


    predictors = prepare_input_features(noise_stft_features)

    predictors = np.reshape(
        predictors, (predictors.shape[0], predictors.shape[1], 1, predictors.shape[2]))
    predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
    print('predictors.shape:', predictors.shape)

    STFTFullyConvolutional = new_model.predict(predictors)
    print(STFTFullyConvolutional.shape)


    def revert_features_to_audio(features, phase, cleanMean=None, cleanStd=None):
        # scale the outpus back to the original range
        if cleanMean and cleanStd:
            features = cleanStd * features + cleanMean

        phase = np.transpose(phase, (1, 0))
        features = np.squeeze(features)

        # features = librosa.db_to_power(features)
        # that fixes the abs() ope previously done
        features = features * np.exp(1j * phase)

        features = np.transpose(features, (1, 0))
        return noiseAudioFeatureExtractor.get_audio_from_stft_spectrogram(features)


    denoisedAudioFullyConvolutional = revert_features_to_audio(
        STFTFullyConvolutional, noisyPhase, mean, std)
    print("Min:", np.min(denoisedAudioFullyConvolutional),
        "Max:", np.max(denoisedAudioFullyConvolutional))
    # ipd.Audio(data=denoisedAudioFullyConvolutional,
    #         rate=fs)  # load a local WAV file
    sf.write(f"{paths_src.OUTPUT_FOLDER_PATH}/{paths_src.OUTPUT_FILE_NAME}", denoisedAudioFullyConvolutional, sr)
    
