import os
import config
import random
import logging
import librosa
import numpy as np
from torch.utils.data import Dataset

class ESC_Dataset(Dataset):
    def __init__(self, dataset, config, eval_mode = False):
        self.dataset = dataset
        self.config = config
        self.eval_mode = eval_mode
        if self.eval_mode:
            self.dataset = self.dataset[self.config.esc_fold]
        else:
            temp = []
            for i in range(len(self.dataset)):
                if i != config.esc_fold:
                    temp += list(self.dataset[i]) 
            self.dataset = temp           
        self.total_size = len(self.dataset)
        self.queue = [*range(self.total_size)]
        logging.info("total dataset size: %d" %(self.total_size))
        if not eval_mode:
            self.generate_queue()

    def generate_queue(self):
        random.shuffle(self.queue)
        logging.info("queue regenerated:%s" %(self.queue[-5:]))


    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        p = self.queue[index]
        data_dict = {
            "audio_name": self.dataset[p]["name"],
            "waveform": np.concatenate((self.dataset[p]["waveform"],self.dataset[p]["waveform"])),
            "real_len": len(self.dataset[p]["waveform"]) * 2,
            "target": self.dataset[p]["target"]
        }
        return data_dict

    def __len__(self):
        return self.total_size


class ESC50Processor:
    def __init__(self, meta_data_path, audio_path, resample_path, savedata_path):
        self.meta_data_path = meta_data_path
        self.audio_path = audio_path
        self.resample_path = resample_path
        self.savedata_path = savedata_path

    def resample(self, sample_rate=320000):
        """Resample all input audio files and save to resample_path."""
        try:
            if not os.path.exists(self.audio_path):
                raise FileNotFoundError(f"Audio path not found: {self.audio_path}")

            audio_list = os.listdir(self.audio_path)
            if not audio_list:
                raise ValueError("❌ No audio files found!")

            logging.info("-------------Resampling ESC-50-------------")
            for f in audio_list:
                full_f = os.path.join(self.audio_path, f)
                resample_f = os.path.join(self.resample_path, f)
                if not os.path.exists(resample_f):
                    os.system(f"sox -V1 {full_f} -r {sample_rate} {resample_f}")
            logging.info("-------------Resampling Complete-------------")

        except Exception as e:
            logging.error(f"❌ Error in resampling: {e}")

    def build(self):
        """Create new dataset from resampoled files."""
        try:
            if not os.path.exists(self.meta_data_path):
                raise FileNotFoundError(
                    f"❌ Metadata file not found: {self.meta_data_path}"
                )

            logging.info("-------------Building Dataset-------------")
            meta = np.loadtxt(
                self.meta_data_path, delimiter=",", dtype="str", skiprows=1
            )
            output_dict = [[] for _ in range(5)]

            for label in meta:
                name, fold, target = label[0], label[1], label[2]
                resampled_file = os.path.join(self.resample_path, name)
                if not os.path.exists(resampled_file):
                    logging.warning(f"⚠️ Warning: File missing {resampled_file}")
                    continue

                try:
                    y, _ = librosa.load(resampled_file, sr=None)
                    output_dict[int(fold) - 1].append({
                        "name": name, 
                        "target": int(target), 
                        "waveform": y
                    })

                except Exception as e:
                    logging.error(f"Failed to process {name}: {e}")

            np.save(self.savedata_path, output_dict)
            logging.info(f"✅ Dataset saved at {self.savedata_path}")
            logging.info("-------------Dataset Build Complete-------------")
        except Exception as e:
            logging.error(print(f"❌ Error in building dataset: {e}"))
