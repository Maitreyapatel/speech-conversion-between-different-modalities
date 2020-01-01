clc; clear all; close all;

load_path_mask = '../results/mask/mcc/';
load_path_normal_mcc = '../dataset/features/US_102/Normal/mcc/';
save_path_mcc = '../results/mcc/';
save_data = '../results/';

filelist1=dir([load_path_normal_mcc,'*.mcc']);
filelist2=dir([load_path_mask,'*.mat']);
x=[];y=[];cx=[];mcd=[];C=[];tr=[];
for index=1:length(filelist1)
    fprintf('Processing %s\n',filelist1(index).name); 
    fid=fopen([load_path_normal_mcc,filelist1(index).name]);
    x=fread(fid,Inf,'float');
    display(length(x));
    cx=reshape(x,40,length(x)/40);
    
    load([load_path_mask,filelist2(index).name]);
    y=foo';

    %fid1=fopen(['/home/maitreya/WHISPER2SPEECH/WHSP2SPCH/gan/rect_150/converted_mcep/',filelist1(index).name]);
    %y=fread(fid1,Inf,'float');
    %y=reshape(y,25,length(y)/25);
    
    mcd(index)=WHSP2SPCHcal_mcd(cx,y);
    fclose(fid); %fclose(fid1);
    
    [a,b]=size(y);
    y=reshape(y,a*b,1);
    y=double(y);
    save([save_path_mcc,filelist1(index).name],'y','-ascii');
end

mn=mean(mcd)
sd=std(mcd)
fid3=fopen([save_data,'MCD.txt'],'a');
fmt='US_102 Mean: %2.2f Std: %2.2f\n';
fprintf(fid3,fmt,mn,sd);                
fclose(fid3); 