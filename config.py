SEED = 42 

workspace = "./workspace" 
# dataset path
DATA_PATH = "ESC-50-master/"
META_DATA_PATH = "meta/esc50.csv" 
AUDIO_PATH = "audio"
AUGMENTED_PATH = "ESC-50-augmented-data"

esc_fold = 0
num_workers = 0
batch_size = 32 # batch size per GPU x GPU number , default is 32 x 4 = 128
max_epochs = 100
learning_rate = 1e-3

mel_bins = 64
window_size = 64
hop_size = 320
sample_rate = 32000
fmin = 50 
fmax = 14000
loss_type = "clip_ce"
dataset_type = "esc50"

debug = False
fl_local = False 
enable_tscam = True 
htsat_attn_heatmap = False
enable_repeat_mode = False
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
crop_size = None # int(clip_samples * 0.5) deprecated\
lr_scheduler_epoch = [10,20,30]
lr_rate = [0.02, 0.05, 0.1]

# resume_checkpoint = "./workspace/ckpt/htsat_audioset_pretrain.ckpt"
resume_checkpoint = None