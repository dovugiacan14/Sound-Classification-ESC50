SEED = 42 

workspace = "./workspace" 
# dataset path
DATA_PATH = "ESC-50-master/"
META_DATA_PATH = "meta/esc50.csv" 
AUDIO_PATH = "audio"
AUGMENTED_PATH = "ESC-50-augmented-data"

esc_fold = 0
num_workers = 3
batch_size = 32 # batch size per GPU x GPU number , default is 32 x 4 = 128

debug = False
exp_name = "exp_htsat_esc_50" # the saved ckpt prefix name of the model

# HTS-AT hyperparameter
htsat_window_size = 8
htsat_spec_size =  256
htsat_patch_size = 4 
htsat_stride = (4, 4)
htsat_num_head = [4,8,16,32]
htsat_dim = 96 
htsat_depth = [2,2,6,2]
classes_num = 50 # esc: 50 | audioset: 527 | scv2: 35
patch_size = (25, 4) # deprecated
crop_size = None # int(clip_samples * 0.5) deprecated

resume_checkpoint = "./workspace/ckpt/htsat_audioset_pretrain.ckpt"