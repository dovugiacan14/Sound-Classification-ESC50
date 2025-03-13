# Sound-Classification-ESC50
## Overview 
This project implements an audio classification system trained on the ESC-50 dataset using deep learning models. The models include:

- **Custom-built CNN architectures.**

- **Transformers.**

- **Bi-Directional-LSTM.**

- **HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER (HTS-AT)** 

## Dataset
The ESC-50 dataset is a collection of 50 classes of environmental sounds, each with 40 samples, making a total of 2000 labeled audio recordings. These sounds include human speech, animal noises, and natural phenomena.

## Requirements
 - Python 3.9+
 - pip 24.0
 - CUDA >= 11.8
 - Pytorch >= 2.0.0
 
## Installation

To set up the project, follow these steps:
1. Clone the repository: 
```bassh
git clone https://github.com/dovugiacan14/Sound-Classification-ESC50 
cd Sound-Classification-ESC50 
```
2. **Install the required packages from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage 

To train and evaluate the model, simply run: 
```bash
python main.py 
```

This script will:

- Load and preprocess the ESC-50 dataset.

- Train the selected deep learning model.

- Evaluate the model’s performance. 

## License 

This project is released under the MIT License.

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## NOTE 
1. Khi chưa có file dataset thì chạy thêm ```python build_dataset.py```