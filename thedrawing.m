function thedrawing(map_aff,tsize,min_angle)

[~,Sz] = size(map_aff);

R = zeros(3,3);
dummy = zeros(1,6);
Outs = [ 1 tsize(1) 1;1 1 tsize(2);1 1 1];
InitPos = zeros(2,3);

for i=1:Sz
    dummy=map_aff{i};
    R = [dummy(1,1) dummy(1,2) dummy(1,5);...
         dummy(1,3) dummy(1,4) dummy(1,6);...
         0          0          1];
    Position = round(R*Outs);
    InitPos = Position(1:2,:);
    %nX = mean([InitPos(2,1) InitPos(2,3)]);
    %nY = mean([InitPos(1,1) InitPos(1,2)]);
    %plot(nX,nY,'*r');hold on
    %plot([InitPos(2,:) InitPos(2,2)+(abs(InitPos(2,1)-InitPos(2,3)))],[InitPos(1,:) InitPos(1,2)],'g');hold on
    YY = [InitPos(2,2) InitPos(2,1) InitPos(2,3) InitPos(2,2)+abs(InitPos(2,3)-InitPos(2,1)) InitPos(2,2)];
    XX = [InitPos(1,2) InitPos(1,1) InitPos(1,3) InitPos(1,2) InitPos(1,2)];
    if i<=11      
        text(mean([InitPos(2,2),InitPos(2,3)]),mean([InitPos(1,2),InitPos(1,3)]),num2str(i),'FontSize',10,'Color','r');
        text(mean([InitPos(2,2),InitPos(2,3)])+15,mean([InitPos(1,2),InitPos(1,3)]),num2str(round(min_angle(i))),'FontSize',10,'Color','b');
        plot(YY,XX,'r');
    else
        text(mean([InitPos(2,2),InitPos(2,3)]),mean([InitPos(1,2),InitPos(1,3)]),num2str(i),'FontSize',10,'Color','g');
        text(mean([InitPos(2,2),InitPos(2,3)])+15,mean([InitPos(1,2),InitPos(1,3)]),num2str(round(min_angle(i))),'FontSize',10,'Color','b');
        plot(YY,XX,'g');
    end
    axis([1 1470 1 661]);
    hold on
end
end