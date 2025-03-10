import os 
import tqdm
import logging 
import librosa
import numpy as np 

def extract_audio_features(dataset): 
    """
    Extracts various audio features using librosa and stores them in the dataset.
    
    Parameters:
        dataset (pandas.DataFrame): DataFrame containing 'files_path' column with audio file paths.
        
    Returns:
        pandas.DataFrame: Updated DataFrame with extracted features.
    """
    features = [
        "zero_crossing_rate",
        "chroma_stft",
        "rmse",
        "spectral_centroid",
        "spectral_bandwidth",
        "beat_per_minute",
        "rolloff",
    ] + [f"mfcc_{i}" for i in range(20)]

    for feature in features:
        dataset[feature] = np.zeros(len(dataset))

    logging.info("Starting to extract audio features...")

    for i in tqdm(range(len(dataset))):
        file_path = dataset['files_path'][i] 
        if os.path.exists(file_path):
            y, sr = librosa.load(file_path, mono=True)
            
            dataset.at[i, 'rmse'] = np.mean(librosa.feature.rms(y=y))
            dataset.at[i, 'chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            dataset.at[i, 'spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            dataset.at[i, 'spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            dataset.at[i, 'rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            dataset.at[i, 'zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            dataset.at[i, 'beat_per_minute'] = tempo
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            for e in range(20):
                dataset.at[i, f'mfcc_{e}'] = np.mean(mfcc[e])
    
    return dataset, features
