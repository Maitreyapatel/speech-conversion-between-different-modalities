
clear all;
clc;
close all;


b = fullfile([strcat('../dataset/features/US_102/Normal/Z.mat')]);
z = load(char(b));

a = (z.Z);
a = a';
[i, j] = size(a);



b = fullfile([strcat('../dataset/features/US_102/Normal/Z_vuv.mat')]);
z = load(char(b));

aa = (z.Z);
aa = aa';
% [i, j] = size(a);


% number of batches rem/rem+1
rem = mod(i, 1000);
n = (i - rem)/1000;
disp(n)

temp = 0;
m = 1;


% create batch of 1000X40
for k=1:n
    Feat = a(m:m+999, 1:40);
    Clean_cent = aa(m:m+999, 1);
    
    %fprintf('m = %i',m)
    %fprintf('   temp = %i\n',temp)

    save(['../dataset/features/US_102/batches/VUV/Batch_',num2str(temp),'.mat'],'Feat', 'Clean_cent');    
    fprintf('Batch_%i created\n',temp);
        
    m = m + 1000;
    temp = temp + 1;
end

% if more than 700 rows are not containing zeros 
if rem>700
    b = zeros((1000 - rem), 40);
	bb = zeros((1000 - rem), 1);
    
	Feat = [a(m:m+rem-1, 1:40); b];
    Clean_cent = [aa(m:m+rem-1, 1); bb];

    k = k + 1;
 
    save(['../dataset/features/US_102/batches/VUV/Batch_',num2str(temp),'.mat'],'Feat','Clean_cent');
    fprintf('Batch_%i created\n',temp);
    
end
