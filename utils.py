import os
import librosa
import logging 
import pandas as pd
import numpy as np 

#  ========================= PART 1 =============================

# df = pd.read_csv('processed_meta.csv')
# df['fold'] = df['fold'].apply(lambda x: np.random.randint(1, 6) if pd.isna(x) else x)

# df.to_csv('data_filled.csv', index=False)

#  ========================= PART 2 =============================
# Đọc dữ liệu
df = pd.read_csv("data_filled.csv")

# Các cột đặc trưng âm thanh
feature_columns = [
    "zero_crossing_rate", "chroma_stft", "rmse", "spectral_centroid",
    "spectral_bandwidth", "beat_per_minute", "rolloff",
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6",
    "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12",
    "mfcc_13", "mfcc_14", "mfcc_15", "mfcc_16", "mfcc_17", "mfcc_18", "mfcc_19"
]

# Khởi tạo danh sách chứa dữ liệu theo từng fold
output_dict = [[] for _ in range(5)]

# Lặp qua từng dòng trong DataFrame để lấy dữ liệu
for _, row in df.iterrows():
    name = os.path.basename(row["files_path"]) 
    fold = int(row["fold"])  
    target = int(row["target"])  
    y = row[feature_columns].values.astype(np.float32)  

    # Lưu vào danh sách tương ứng với fold
    output_dict[fold - 1].append({"name": name, "target": target, "waveform": y})

# Chuẩn hóa số lượng mẫu về 3200 mỗi fold
TARGET_SIZE = 3200
for i in range(5):
    current_size = len(output_dict[i])

    if current_size < TARGET_SIZE:
        # 🔹 Bổ sung dữ liệu bằng cách lặp lại ngẫu nhiên
        while len(output_dict[i]) < TARGET_SIZE:
            output_dict[i].append(output_dict[i][np.random.randint(0, current_size)])

    elif current_size > TARGET_SIZE:
        # 🔹 Cắt bớt dữ liệu
        output_dict[i] = output_dict[i][:TARGET_SIZE]

    print(f"✅ Fold {i+1} có {len(output_dict[i])} phần tử sau chuẩn hóa")

# Lưu dữ liệu thành file .npy
np.save("dataset.npy", np.array(output_dict, dtype=object), allow_pickle=True)
