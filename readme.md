# Whisper-to-Normal Speech Conversion using Generative Adversarial Networks
____ 


### Tasks:

- [x] Feature extraction
- [x] Python scripts for MCC training
- [x] Python scripts for F0 training
- [x] Python scripts for VUV training
- [x] Matlab scripts for objective measures
- [x] Synthesis using Ahocoder
- [x] Single shell script for complete automation
- [x] Check the the whole repository for reproducibility.
- [x] Make the whole documentation.
- [ ] Add python scripts for DiscoGAN, CycleGAN, DC-GAN, Inception-GAN.

## Prerequisites:

- Linux, MacOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- SoX (Command line utility that can convert various formats of computer audio files in to other formats.)
- MatLab


### Before going ahead please create the two directories (dataset and results) with follwing sctructre:

```
dataset
|
└───features
    └───US_102
        └───batches
        |   |    f0
        |   |    mcc
        |   |    VUV
        |
        └───Normal
        |   |    f0
        |   |    mcc
        |   |    fv
        |
        └───Whisper
            |    f0
            |    mcc
            |    fv
```
```
results
|
└───checkpoints
|   |   mcc
|   |   f0
|   |   vuv
|
└───mask
|   |   mcc
|   |   f0
|   |   vuv
|
└───converted_f0
|
└───converted_wav
|
└───mcc
|
└───tmp

```

## Required Python libraries:

```
torchvision==0.2.1
visdom==0.1.8.8
matplotlib==3.1.2
numpy==1.17.4
scipy==1.3.2
torch==1.3.1
```
These libraries can be installed via following command:

```
pip install -r requirements.txt
```


## Dataset:
For whisper-to-normal speech conversion, we rely on wTIMIT dataset. Please download this dataset from [here](http://www.isle.illinois.edu/sst/data/wTIMIT/). And follow below steps:

1. Download the dataset put the .wav files in the `dataset/data/{speaker_name}/{Normal/Whisper}`.
2. In this GitHub repository we use Ahocoder to extract MCEP/MCC features. However, Ahocoder accepts input .wav files with 16,000 Hz frequency only. Thereofore, run following two terminal commands to each .wav files.
    - `sox {path + file1.wav} -t -wav {path + file1.wav}` (Only if downloaded files has .WAV extention.)
    - `sox {path + file1.wav} -r 16000 -c 1 -b 16 {path + file1.wav}`
    
# Now, do the training for WHSP2SPCH conversion:

Either run the follwing single shell script or follow the steps described below:

```
./run.sh
```

**OR Follow these steps**


## Feature extraction:

Now, run follwing commads in the terminal:

1. `cd scripts`
2. `./feature_extraction.sh`

## Training Batches creation:

Run below mentioned MatLab scripts in **order**.
- concatenation_VCC2018.m
- batch.m
- concatenation_f0.m
- batch_f0.m
- concatenation_vuv.m
- batch_vuv.m

## Training of Deep Learning mapping function:

Yay! Finally it's time to do the training. Now, go to `py_src` directory and run below mentioned commads in terminal.

1. `python -m visdom.server`
2. `python MMSE_GAN.py -tr -te`
3. `python MMSE_GAN_F0.py -tr -te`
4. `python DNN_vuv.py -tr -te`

**Note:**  These python scripts are highly customizable according to the user choises. Such as traininng/testing, parallel/non-parallel training, paths to the features/checkpoint, learning rate, number of epoch, validation interval, check point interval etc.

## Obective measures:

As decribed in the research papers, we use MCD and F0_RMSE as objective evalution metrc. Hence, run follwing two MatLab scripts:

- Testing_MCD.m
- f0obective_cal.m

## Speech Synthesis:

Finally, we are at the last stage. We will use converted mcc and predicted F0 feature as an input to Ahocoder and it will generate the speech from these features. Hence, run follwing command in terminal (scripts directory):

1. `./synthesis.sh`

## Reference Papers: 

- **Novel Inception-GAN for Whisper-to-Normal Speech Conversion** [[Link]](https://www.isca-speech.org/archive/SSW_2019/abstracts/SSW10_P_1-9.html) 

- **Effectiveness of Cross-Domain Architectures for Whisper-to-Normal Speech Conversion** [[Link]](https://ieeexplore.ieee.org/abstract/document/8902961)

- **Novel mmse discogan for cross-domain whisper-to-speech conversion** [[Link]](https://drive.google.com/file/d/1UVbXRzpaM1_ayaTfvq92RVijn_M8FLnn/view)

## Citation:

If you use this code for your research, please cite our papers.

```
@inproceedings{parmar2019effectiveness,
  title={Effectiveness of Cross-Domain Architectures for Whisper-to-Normal Speech Conversion},
  author={Parmar, Mihir and Doshi, Savan and Shah, Nirmesh J and Patel, Maitreya and Patil, Hemant A},
  booktitle={2019 27th European Signal Processing Conference (EUSIPCO)},
  pages={1--5},
  year={2019},
  organization={IEEE}
}

@inproceedings{Patel2019,
  author={Maitreya Patel and Mihir Parmar and Savan Doshi and Nirmesh Shah and Hemant Patil},
  title={{Novel Inception-GAN for Whispered-to-Normal Speech Conversion}},
  year=2019,
  booktitle={Proc. 10th ISCA Speech Synthesis Workshop},
  pages={87--92},
  doi={10.21437/SSW.2019-16},
  url={http://dx.doi.org/10.21437/SSW.2019-16}
}
```
