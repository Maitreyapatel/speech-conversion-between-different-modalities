#!/bin/sh
for entry in `ls ../Results/mcc/*.mcc`;do
fname=`basename $entry .mcc` 
echo $fname
x2x +af < $entry > ../Results/tmp/$fname.mcc
x2x +af < ../Results/converted_f0/$fname.f0 > ../Results/tmp/$fname.f0
./ahodecoder16_64 ../Results/tmp/$fname.f0 ../Results/tmp/$fname.mcc ../Results/converted_wav/$fname.wav
rm -r ../Results/tmp/*
done
exit
