clc, clear all, close all;

load ../dataset/ripley
addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

% data visualization
figure('Color',[1 1 1]);
gscatter(Xt(:,1),Xt(:,2), Yt, 'br', 'ox');
xlabel('X'); ylabel('Y'); title('Scatterplot');

export_fig('ripley_scatter.pdf')

figure('Color',[1 1 1]);
gscatter(X(:,1),X(:,2), Y, 'br', 'ox');
xlabel('X'); ylabel('Y'); title('Scatterplot');

% Linear model
gamlist=logspace(-4,3); type='c'; errlist=[];

for gam=gamlist,
    [alpha,b] = trainlssvm({Xt,Yt,type,gam,[],'lin_kernel'});
    [Yht, Ylin] = simlssvm({Xt,Yt,type,gam,[],'lin_kernel'}, {alpha,b}, X);
    err = sum(Yht~=Y); errlist = [errlist; err];
end
[min, index] = min(errlist);
figure('Color',[1 1 1]);
subplot(1,2,1);
plotlssvm({Xt,Yt,type,gamlist(index),[],'lin_kernel','preprocess'},{alpha,b});
hold on;
[Yht, Ylin] = simlssvm({Xt,Yt,type,gam,[],'lin_kernel'}, {alpha,b}, X);

export_fig('ripley_linear.pdf')

% RBF with tunning 
model_csa = {Xt, Yt, 'c', [], [], 'RBF_kernel', 'csa'};
[gam, sig2, cost] = tunelssvm(model_csa, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});

[alpha,b] = trainlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'});
[estYval, YRBF] = simlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'}, {alpha,b}, X);
err = sum(estYval~=Y);
subplot(1,2,2);
plotlssvm({Xt,Yt,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

export_fig('ripley_lin_RBF.pdf')

% ROC curve
figure('Color',[1 1 1]);
roc(Ylin, Y);
hold on;
roc(YRBF, Y);

export_fig('ripley_ROC.pdf')
