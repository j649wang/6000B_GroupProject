file=dir(strcat('./data'));
for i=3:length(file)
% for i=3:5
    i
    im=imread(strcat('./data/',file(i).name));
    if file(i).name(7)=='r'
      im=fliplr(im);
    end
    level=graythresh(im);
    im1=imbinarize(im,level);
    [x1,y1]=find(im1,1,'first');
    [x2,y2]=find(im1',1,'first');
    [x3 y3]=find(im1,1,'last');
    [x4 y4]=find(im1',1,'last');
    im2=im(y2:y4,y1:y3);
    im3=imresize(im2,[64,64]);
%     im3=imrotate(im3, 270);
%     im3=flipud(im3);
    imwrite(im3,['./dataresize64B/' file(i).name(1:8) '.png']);
end 
    
