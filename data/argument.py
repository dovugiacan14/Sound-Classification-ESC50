import os
import tqdm
import librosa
import numpy as np
import soundfile as sf


class DataArgumentor:
    def __init__(self, dataset, output_file):
        self.dataset = dataset
        self.output_file = output_file
        os.makedirs(output_file, exist_ok=True)

    def add_noise(data):
        """
        Adds Gaussian noise to the audio data.

        Parameters:
        - data (numpy.ndarray): The input audio signal.

        Returns:
        - numpy.ndarray: The audio signal with added noise.
        """
        return data + np.random.normal(0, 0.002, len(data))

    def pitch_shifting(data, sr=16000):
        """
        Shifts the pitch of the audio signal by a random number of semitones.

        Parameters:
        - data (numpy.ndarray): The input audio signal.
        - sr (int, optional): The sample rate of the audio. Default is 16,000 Hz.

        Returns:
        - numpy.ndarray: The pitch-shifted audio signal.
        """
        return librosa.effects.pitch_shift(
            data.astype(np.float32), sr=sr, n_steps=np.random.uniform(-2, 2)
        )

    def time_stretching(data, rate=1.5):
        """
        Stretches or compresses the audio signal in time without altering the pitch.

        Parameters:
        - data (numpy.ndarray): The input audio signal.
        - rate (float, optional): The time-stretch factor. Values greater than 1 speed up the audio,
          while values less than 1 slow it down. Default is 1.5.

        Returns:
        - numpy.ndarray: The time-stretched audio signal.
        """
        return librosa.effects.time_stretch(data.astype(np.float32), rate=rate)

    def volume_change(data, gain=3):
        """
        Modifies the volume of the audio signal.

        Parameters:
        - data (numpy.ndarray): The input audio signal.
        - gain (float, optional): The gain change in decibels (dB). Positive values increase the volume,
          while negative values decrease it. Default is 3 dB.

        Returns:
        - numpy.ndarray: The volume-adjusted audio signal.
        """
        return data * (10 ** (gain / 20))

    def save_augmented_audio(self, filename, data, sr=16000):
        sf.write(filename, data, sr)

    def process(self):
        augmented_files = []
        augmented_targets = []
        for _, row in tqdm(self.dataset.iterrows(), total=len(self.dataset)):
            file_path = row["files_path"]
            label = row["target"]

            try:
                data, sr = librosa.load(file_path, sr=16000)
                augmentations = [
                    self.add_noise(data),
                    self.pitch_shifting(data, sr=sr),
                    self.time_stretching(data),
                    self.volume_change(data),
                ]

                for idx, aug_data in enumerate(augmentations):
                    aug_filename = f"{row['filename'][:-4]}_aug_{idx}.wav"
                    aug_filepath = os.path.join(self.output_file, aug_filename)
                    self.save_augmented_audio(aug_filepath, aug_data, sr)

                    augmented_files.append(aug_filepath)
                    augmented_targets.append(label)
                return augmented_files, augmented_targets

            except Exception as e:
                print(f"⚠️ Error occured when argumenting data for {file_path}: {e}")
                return [], []
