clear
clc
image=imread('face.jpg');%读取面部图像，大小为128×128
double_f=zeros(128,128);
lookup=zeros(1,256);%查找表
for i=1:256
    lookup(i)=log(i);%定义查找表
end
[row col dim]=size(image);
for i=1:row
    for j=1:col
    double_f(i,j)=lookup(image(i,j)+1);
    end
end
tic;
dft_image=fft2(double_f);%面部图像的dft

%左眼dft
left_filter=load('left_filter.txt');
left_filter_dft=fft2(left_filter);
%右眼dft
right_filter=load('right_filter.txt');
right_filter_dft=real(fft2(right_filter));
%左眼滤波结果
left_filter_reslut=real(dft_image.*left_filter_dft);
%右眼滤波结果
right_filter_result=real(dft_image.*right_filter_dft);

%左眼滤波结果转换到空域
left_filter_ifft=real(ifft2(dft_image.*left_filter_dft));

%右眼滤波结果转换到空域
right_filter_ifft=ifft2(dft_image.*right_filter_dft);

%查找右眼的中心点
left_eye_region=left_filter_ifft(35:66,23:54);
result=find(left_eye_region==max(left_eye_region(:)));
left_eye_x=mod(result,32)+23;
left_eye_y=rem(result,32)+35;
%查找左眼的中心点
right_eye_region=right_filter_ifft(32:63,71:102);
result=find(right_eye_region==max(right_eye_region(:)));

right_eye_x=mod(result,32)+71;
right_eye_y=rem(result,32)+32;
imshow(image)


