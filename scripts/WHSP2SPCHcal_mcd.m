function z=NAM2WHSPcal_mcd(X,Y)
         [min_distance,d,g,path] = dtw_E(X,Y);
         z=mean(((sqrt(2)*10)/log(10))*(sqrt(sum((X(2:40,path(:,1))-Y(2:40,path(:,2))).^2))));
         %z=mean(((sqrt(2)*10)/log(10))*(sqrt(sum((X-Y)).^2)));
end
