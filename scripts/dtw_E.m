function [min_distance, d, g, path] = dtw_E(A, B)
% Minimal time normalized dtw distance between speech patterns A and B.

% References:
%
% [SakoeChiba1978] SAKOE, Hiroki; CHIBA, Seibi: Dynamic Programming
%     Algorithm Optimization for Spoken Word Recognition,
%     http://citeseer.ist.psu.edu/viewdoc/download?doi=10.1.1.114.3782&rep=rep1&type=pdf
%
% % [Ellis2003] ELLIS, D.: Dynamic Time Warp (DTW) in Matlab,
%     http://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/
% Modified by Nirmesh J Shah, DA-IICT, Gandhinagar 
% P is the distance function

% get length of speech patterns A and B
[~, I] = size(A);
[~, J] = size(B);

r=abs(I-J);

% local distance matrix
% d = zeros(I, J);
% for i = 1:I
%     for j = 1:J
%         d(i,j) = my_distance(A(:,i),B(:,j),'c');
%     end
% end

d = pdist2(A',B');
d=single(d);
 % global distance matrix
g = zeros(I+1,J+1);
g(:,:) = inf;
g(1,1) = 2*d(1,1); % initial condition, see (19) in [SakoeChiba1978]

% if you want to set s =1 then check the value of r for [SakoeChiba]
steps = zeros(I,J); % steps to take in order to reach D(i,j)

for i = 2:I+1;
    for j = 2:J+1;
        if (abs(i-j) > r)
            % we're outside the adjustment window
            continue;
        end

        % local distance matrix is smaller than g, translate coordinates
        i_l = i-1;
        j_l = j-1;

        % calculate global distances
        % (see DP-equation (20) from [SakoeChiba1978] for reference)
        [distance, step] =  min([g(i, j-1) +   d(i_l, j_l)...
                                 g(i-1, j-1) + 2*d(i_l, j_l)...
                                 g(i-1, j)   +   d(i_l, j_l)]);
        g(i,j) = distance;
        steps(i-1,j-1) = step;
    end
end

% time normalize global distance matrix
N=I+J;
D=g/N;

% remove additional inf padded row and column from global distance matrix
D=D(2:end,2:end);

path=traceback_path(steps);

min_distance = D(end, end);


