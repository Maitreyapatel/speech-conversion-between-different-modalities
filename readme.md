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
- [ ] **Check the the whole repository for reproducibility.**
- [x] Make the whole documentation.
- [ ] Add python scripts for DiscoGAN, CycleGAN, DC-GAN, Inception-GAN.



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


## Dataset:
For whisper-to-normal speech conversion, we rely on wTIMIT dataset. Please download this dataset from [here](http://www.isle.illinois.edu/sst/data/wTIMIT/). And follow below steps:

1. Download the dataset put the .wav files in the `dataset/data/{speaker_name}/{Normal/Whisper}`.
2. In this GitHub repository we use Ahocoder to extract MCEP/MCC features. However, Ahocoder accepts input .wav files with 16,000 Hz frequency only. Thereofore, run following two terminal commands to each .wav files.
    - `sox {path + file1.wav} -t -wav {path + file1.wav}` (Only if downloaded files has .WAV extention.)
    - `sox {path + file1.wav} -r 16000 -c 1 -b 16 {path + file1.wav}`
    
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

## Implementation of the following research papers: 

- **Novel Inception-GAN for Whisper-to-Normal Speech Conversion** [[Link]](https://www.isca-speech.org/archive/SSW_2019/abstracts/SSW10_P_1-9.html) 

- **Effectiveness of Cross-Domain Architectures for Whisper-to-Normal Speech Conversion** [[Link]](https://ieeexplore.ieee.org/abstract/document/8902961)
