% % % Saving Features with and without Outliers for different configurations

clc;
clear all; 
close all;


filelist=dir(['../dataset/features/US_102/Normal/mcc/*.mcc']);
filelist1=dir(['../dataset/features/US_102/Whisper/mcc/*.mcc']);

dim=40;result=[];mpsrc=[];mptgt=[];t_scores=[];l=[];lz=[];wr=[];mpsrc1=[];mptgt1=[];Z1=[];Z=[];Z2=[];
x=[];y=[];X=[];Y=[];path=[];

for index=1:length(filelist)

    fprintf('Processing %s\n',filelist(index).name);

    fid=fopen(['../dataset/features/US_102/Normal/mcc/',filelist(index).name]);
    x=fread(fid,Inf,'float');
    x=reshape(x,dim,length(x)/dim);

    fid1=fopen(['../dataset/features/US_102/Whisper/mcc/',filelist1(index).name]);
    y=fread(fid1,Inf,'float');
    y=reshape(y,dim,length(y)/dim);

    [min_distance, d, g, path] = dtw_E(x, y);
    x = x(1:40,path(:,1));
    y = y(1:40,path(:,2));
    X=[X x];
    Y=[Y y];

    fclose('all');

end

Z=[X];
save(['../dataset/features/US_102/Normal/Z.mat'],'Z'); 
Z=[Y];
save(['../dataset/features/US_102/Whisper/Z.mat'],'Z');



