#!/bin/sh
for entry in `ls ../results/mcc/*.mcc`;do
fname=`basename $entry .mcc` 
echo $fname
x2x +af < $entry > ../results/tmp/$fname.mcc
x2x +af < ../results/converted_f0/$fname.f0 > ../results/tmp/$fname.f0
./ahodecoder16_64 ../results/tmp/$fname.f0 ../results/tmp/$fname.mcc ../results/converted_wav/$fname.wav
rm -r ../results/tmp/*
done
exit
