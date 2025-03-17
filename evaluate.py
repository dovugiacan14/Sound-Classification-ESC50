import os
import torch
import config
import librosa
import argparse
import numpy as np
from dotenv import load_dotenv
from models.htsat import HTSAT_Swin_Transformer

load_dotenv()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sound Classification Evaluate.")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model",
    )

    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio to predict",
    )
    return parser.parse_args()


def predict(model_path, audio, gd):
    Audio_cls = Audio_Classification(model_path, config)
    pred_label, pred_prob = Audio_cls.predict(audio)
    print("--- Result Prediction ---")
    print(f"Audiocls predict output: {pred_label} with probability: {pred_prob}")
    print(f"Actual class: {gd}")


class Audio_Classification:
    def __init__(self, model_path, config):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config=config,
            depths=config.htsat_depth,
            embed_dim=config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head,
        )
        ckpt = torch.load(model_path, map_location="cpu")
        temp_ckpt = {}
        for key in ckpt["state_dict"]:
            temp_ckpt[key[10:]] = ckpt["state_dict"][key]
        self.sed_model.load_state_dict(temp_ckpt)
        self.sed_model.to(self.device)
        self.sed_model.eval()

    def predict(self, audiofile):
        if audiofile:
            waveform, sr = librosa.load(audiofile, sr=32000)

            with torch.no_grad():
                x = torch.from_numpy(waveform).float().to(self.device)
                output_dict = self.sed_model(x[None, :], None, True)
                pred = output_dict["clipwise_output"]
                pred_post = pred[0].detach().cpu().numpy()
                pred_label = np.argmax(pred_post)
                pred_prob = np.max(pred_post)
            return pred_label, pred_prob

if __name__ == "__main__": 
    args = parse_arguments() 

    DATA_PATH = os.environ.get("DATA_PATH")
    META_PATH = os.environ.get("META_DATA_PATH")

    meta_path = os.path.join(DATA_PATH, META_PATH)
    meta = np.loadtxt(meta_path, delimiter=",", dtype="str", skiprows=1)

    gd = {}
    for label in meta:
        name = label[0]
        target = label[2]
        gd[name] = target
    
    audio_file = args.audio.split("/")[-1]
    predict(
        model_path= args.model, 
        audio= args.audio, 
        gd = gd[audio_file]
    )


