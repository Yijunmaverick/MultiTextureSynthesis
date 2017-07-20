outputVideo = VideoWriter('XXX.avi');
outputVideo.FrameRate =24;
open(outputVideo)
  
for i = 1:500 
   sname = strcat(num2str(i),'.jpg');
   img=imread(strcat('./', sname));
   writeVideo(outputVideo,uint8(img));
end

close(outputVideo)
