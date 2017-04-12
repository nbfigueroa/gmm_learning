function plot_data(X,labels)
% utility function to plot the data based on the first 2 principle
% components of the data.

pc = princomp(X);

if nargin > 1
    gscatter(X*pc(:,1),X*pc(:,2),labels);
else
    gscatter(X*pc(:,1),X*pc(:,2));
end