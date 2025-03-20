import os
import dotenv
import torch
import config
import argparse
import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from helpers.constant import decoder
from helpers.utils import extract_features
from models.htsat import HTSAT_Swin_Transformer

load_dotenv()
DATA_PATH = os.environ.get("DATA_PATH")
META_PATH = os.environ.get("META_DATA_PATH")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_label(audio_file):
    audio_filename = os.path.basename(audio_file)
    meta_path = os.path.join(DATA_PATH, META_PATH)
    meta = np.loadtxt(meta_path, delimiter=",", dtype="str", skiprows=1)
    gd = {}
    for label in meta:
        name = label[0]
        target = label[2]
        gd[name] = target
    encoded_label = str(gd[audio_filename])
    # return decoder[int(encoded_label)]
    return encoded_label


def predict(audio, model, model_type, target_size=32):
    features = extract_features(audio)

    # if .h5 model 
    if model_type == "h5": 
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        pred_label = np.argmax(prediction)

    # if ckpt model 
    elif model_type == "ckpt": 
        if len(features) < target_size:
            padding = np.zeros(target_size - len(features))
            features = np.concatenate((features, padding))

        with torch.no_grad():
            x = torch.from_numpy(features).float().to(device)
            x = x.unsqueeze(0) 
            # predict 
            output_dict = model(x) 
            pred =  output_dict["clipwise_output"]
            pred_post = pred[0].detach().cpu().numpy()
            pred_label = np.argmax(pred_post)

    # return decoder[pred_label]
    return pred_label

def load_trained_model(model_path):
    model_filename = os.path.basename(model_path)
    if ".h5" in model_filename:
        model = load_model(model_path)
        return model, "h5"

    elif ".ckpt" in model_filename:
        model = HTSAT_Swin_Transformer(
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
        temp_ckpt = {key[10:]: ckpt["state_dict"][key] for key in ckpt["state_dict"]}
        model.load_state_dict(temp_ckpt)

        model.to(device)
        return model.eval(), "ckpt"

    else:
        raise ValueError(f"Unsupported model format: {model_filename}")


if __name__ == "__main__":
    args = parse_arguments()
    audio_path = args.audio.replace("\\", "/")
    true_label = load_label(audio_path)
    model, model_type = load_trained_model(args.model)
    predicted_label = predict(audio_path, model, model_type)
    print(f"🔮 Predicted Label: {predicted_label}")
    print(f"🎯 True Label: {true_label}")
