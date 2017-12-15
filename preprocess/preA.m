file=dir(strcat('./data'));
for i=3:length(file)
% for i=3: 4
    i
    im=imread(strcat('./data/',file(i).name));
    if strfind(file(i).name,'_L_')==29
      im=fliplr(im);
    end
    level=graythresh(im);
    im1=imbinarize(im,level);
    [x1,y1]=find(im1,1,'first');
    [x2,y2]=find(im1,1,'last');
    im2=im(1:x2,y1:size(im,2));
    im3=imresize(im2,[256,256]);
% for j=-5:5
%     j
%     im1=imrotate(im,5*j);
    imwrite(im3,['./dataresize128A/' file(i).name(1:8) '.png']);
% end
end
    
