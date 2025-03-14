import os
from dotenv import load_dotenv
from helpers.utils import create_path
from helpers.data_processor import ESC50Processor

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", None)
META_DATA_PATH = os.path.join(DATA_PATH, os.getenv("META_DATA_PATH", None))
AUDIO_PATH = os.path.join(DATA_PATH, os.getenv("AUDIO_PATH", None))

RESAMPLE_PATH = os.path.join(DATA_PATH, "resample")
SAVEDATA_PATH = os.path.join(DATA_PATH, "esc-50-data.npy")

create_path(RESAMPLE_PATH)

if __name__ == "__main__":
    data_processor = ESC50Processor(
        meta_data_path=META_DATA_PATH,
        audio_path=AUDIO_PATH,
        resample_path=RESAMPLE_PATH,
        savedata_path=SAVEDATA_PATH,
    )
    data_processor.resample()
    data_processor.build()
