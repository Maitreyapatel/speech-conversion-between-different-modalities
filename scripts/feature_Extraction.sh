#!/bin/sh
for entry1 in `ls ../dataset/data/US_102/Normal/*.wav`;do
fname=`basename $entry1 .wav`
echo $fname 
./ahocoder16_64 "../dataset/data/US_102/Normal/$fname.wav" "../dataset/features/US_102/Normal/f0/$fname.f0" "../dataset/features/US_102/Normal/mcc/$fname.mcc" "../dataset/features/US_102/Normal/fv/$fname.fv"
done


#!/bin/sh
for entry1 in `ls ../dataset/data/US_102/Whisper/*.wav`;do
fname=`basename $entry1 .wav`
echo $fname 
./ahocoder16_64 "../dataset/data/US_102/Whisper/$fname.wav" "../dataset/features/US_102/Whisper/f0/$fname.f0" "../dataset/features/US_102/Whisper/mcc/$fname.mcc" "../dataset/features/US_102/Whisper/fv/$fname.fv"
done
exit
