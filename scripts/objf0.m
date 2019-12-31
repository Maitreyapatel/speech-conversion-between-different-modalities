function [rms,vu]=objf0(x,y)
         % columnwise in one row
         cx=x; cy=y;
         x(x>0)=1; x(x<0)=0;
         y(y>0)=1; y(y<0)=0;
         z=x+y;
         in=find(z==2);
         dx=cx(in); dy=cy(in);
         rms=sqrt(sum((dx-dy).^2));
         inv=find((x-y)==0);
         vu=(length(inv)/length(x))*100;
end