
# Change the directory
cd ./scripts

# Run feature extraction
./feature_Extraction.sh

# Run MatLab scripts for generating training batches
## For MCEP features
matlab -nodisplay -nosplash -nodesktop -r "run('/home/speechlab/Maitreya/whsp2spch/whisper-to-normal-speech-conversion/scripts/concatenation_VCC2018.m');exit;" | tail -n +11

matlab -nodisplay -nosplash -nodesktop -r "run('/home/speechlab/Maitreya/whsp2spch/whisper-to-normal-speech-conversion/scripts/batch.m');exit;" | tail -n +11


## For F0 features
matlab -nodisplay -nosplash -nodesktop -r "run('/home/speechlab/Maitreya/whsp2spch/whisper-to-normal-speech-conversion/scripts/concatenation_f0.m');exit;" | tail -n +11

matlab -nodisplay -nosplash -nodesktop -r "run('/home/speechlab/Maitreya/whsp2spch/whisper-to-normal-speech-conversion/scripts/batch_f0.m');exit;" | tail -n +11


## For VUV features
matlab -nodisplay -nosplash -nodesktop -r "run('/home/speechlab/Maitreya/whsp2spch/whisper-to-normal-speech-conversion/scripts/concatenation_vuv.m');exit;" | tail -n +11

matlab -nodisplay -nosplash -nodesktop -r "run('/home/speechlab/Maitreya/whsp2spch/whisper-to-normal-speech-conversion/scripts/batch_vuv.m');exit;" | tail -n +11


# Change the directory
cd ../py_src

# Run python scripts for MCEP, F0, and VUV features
python3 MMSE_GAN.py -tr -te
python3 MMSE_GAN_F0.py -tr -te
python3 DNN_VUV.py -tr -te


# Change the directory
cd ../scripts

# Do the objective evaluations
matlab -nodisplay -nosplash -nodesktop -r "run('/home/speechlab/Maitreya/whsp2spch/whisper-to-normal-speech-conversion/scripts/Testing_MCD.m');exit;" | tail -n +11

matlab -nodisplay -nosplash -nodesktop -r "run('/home/speechlab/Maitreya/whsp2spch/whisper-to-normal-speech-conversion/scripts/f0objective_cal.m');exit;" | tail -n +11


# Now do the speech synthesis
./synthesis.sh
