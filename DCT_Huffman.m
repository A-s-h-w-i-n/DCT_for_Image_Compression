clc
clear all
close all

%--------COMPRESSION---------

file = uigetfile
[filepath,name,ext] = fileparts(file);
[inImage,colormap] = imread(strcat(name,ext));
C = strsplit(name,'_');
figure(1)
inImage1 = inImage(:,:,1);
imshow(inImage1);
title('Input Image');

size_img = size(inImage1)

A1 = double(inImage1);



%---------------DCT Matrix for N=8----------------
N = 8;              
for k = 0:N-1
    for n = 0:N-1
        if(k==0)
            a = sqrt(1/N);
        else
            a = sqrt(2/N);
        end
        
        X(k+1,n+1) = a*cos(((2*n+1)*pi*k)/(2*N));
        
    end
end

q = 0.01;

for x=1:8:size_img(1)-rem(size_img(1),8)
    for y=1:8:size_img(2)-rem(size_img(2),8)
        img(1:8,1:8) = A1(x:x+7,y:y+7);   %Block transformation approach             
                                               
        trans_img8 = X*img*X';                  % DCT   
        
        mask = [16 11  10 16 24 40 51 61;
                12 12 14 19 26 58 60 55;
                14 13 16 24 40 57 69 56;
                14 17 22 29 51 87 80 62;
                18 22 37 56 68 109 103 77;
                24 35 55 64 81 104 113 92;
                49 64 78 87 103 121 120 101;
                72 92 95 98 112 100 103 99];
            
        trans_img8 = floor(trans_img8./(mask*q) + 0.5);
        A1(x:x+7,y:y+7) = trans_img8;
    end
end


figure(2)
imshow(uint8(A1));
title('DCT Transformed Image');

MIN = min(min(A1));
MAX = max(max(A1));
A = A1 - MIN;
A = floor((A*255)/(MAX-MIN));

symbols = 0:255;

for i = 1:256
    P(i) = 0;
end

for i=1:size_img(1)*size_img(2)
    P(A(i)+ 1) = P(A(i)+ 1) + 1;
end

P = P/(size_img(1)*size_img(2));

for i=1:size_img(1)
    for j=1:size_img(2)
        imageVector(j+(i-1)*size_img(2)) = A(i,j);
    end
end
imageVector = transpose(imageVector);

dict = huffmandict(symbols,P);
for i=1:size_img(1)
    for j=1:size_img(2)
        if(i==1&&j==1)
            comp = dict{A(i,j)+1,2};
        else
            comp = [comp dict{A(i,j)+1,2}];
        end
    end
end

binarySig = de2bi(imageVector);
seqLen = size(binarySig);

compLen = size(comp);

[compBinaryMat,paddedBin] = vec2mat(comp,8);

compDecimalVec = bi2de(compBinaryMat);

[compDecimalMat,paddedDec] = vec2mat(compDecimalVec,size_img(2));

disp('Images compressed successfully \n\n')

decodedImgVec = huffmandeco(comp,dict);
decodedImg = reshape(decodedImgVec,size_img(2),size_img(1))';

Ax = (A*(MAX-MIN)/255);
Ax = Ax + MIN;

for x=1:8:size_img(1)-rem(size_img(1),8)
    for y=1:8:size_img(2)-rem(size_img(2),8)
        img(1:8,1:8) = Ax(x:x+7,y:y+7);   %Block transformation approach                                                                    
        
        mask = [16 11  10 16 24 40 51 61;
                12 12 14 19 26 58 60 55;
                14 13 16 24 40 57 69 56;
                14 17 22 29 51 87 80 62;
                18 22 37 56 68 109 103 77;
                24 35 55 64 81 104 113 92;
                49 64 78 87 103 121 120 101;
                72 92 95 98 112 100 103 99];
            
        trans_img8 = img.*(mask*q);                
        IDCT = X'*trans_img8*X;       
        Ax(x:x+7,y:y+7) = IDCT;
    end
end

figure(3)
imshow(uint8(Ax));
title('Recovered Image');

original_total_bits = size_img(1)*size_img(2)*8;
disp('Total bits before compression: ')
disp(original_total_bits)

compSize = size(uint8(compDecimalMat));
compressed_total_bits = compSize(1)*compSize(2)*8;
disp('Total bits after compression: ')
disp(compressed_total_bits)

Compression_Ratio = original_total_bits/compressed_total_bits;
disp('Compression ratio: ')
disp(Compression_Ratio)

Space_Saving = 1 - 1/Compression_Ratio;
disp('Space Saving: ')
disp(Space_Saving)  

L_avg = 0;
for i=1:256
    L_avg = L_avg + length(dict{i,2})*P(i);
end
disp('L average: ')
disp(L_avg)

H = 0;
for i = 1:256
    if(P(i)>0)        
        H = H + P(i)*log(1/P(i))/log(2);
    end
end
disp('Entropy: ')
disp(H)

%efficiency = H*100/L_avg;
%fprintf('Code efficiency: %f%% \r\n',efficiency)

MSE = immse(inImage1, uint8(Ax));
disp('Mean Squared Error: ')
disp(MSE)

PSNR = psnr(inImage1, uint8(Ax));
disp('Signal to Noise Ratio: ')
disp(PSNR)