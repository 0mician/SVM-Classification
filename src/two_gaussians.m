clear; clc; close all;
addpath export_fig

% dataset
X1 = 1 + randn(50,2);
X2 = -1 + randn(51,2);

x_classifier = -4:4; y_classifier = -x_classifier;

% class labels
Y1 = ones(50,1);
Y2 = -ones(51,1);

X = [X1; X2];
Y = [Y1; Y2];

% plotting
figure('Color',[1 1 1]);
hold on;
plot(X1(:,1), X1(:,2),'ro'); plot(X2(:,1), X2(:,2),'bo');
plot(x_classifier, y_classifier, 'k');

title('Two gaussians classification','FontSize',20);
xlabel('X','FontSize',14);
ylabel('Y','FontSize',14);
h_legend = legend('Class 1','Class 2', 'Classifier');
set(h_legend,'FontSize',14);
hold off;

export_fig('two_gaussians.pdf')