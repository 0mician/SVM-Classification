clear all; clc; close all;

load ../dataset/iris
addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

% Create training and validation sets using random indices idx
idx = randperm(size(X,1));
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80),:);
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100),:);

% Searching sigma space for a good value (good generalization on test set)
sig2list=logspace(-3,3,60); errsiglist=[]; gam = 1;
for sig2=sig2list,
    [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
    % Obtain classification of test set using trained classifier
    estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha,b}, Xval);
    err = sum(estYval~=Yval); errsiglist = [errsiglist; err]; 
end

% Searching sigma space for a good value (good generalization on test set)
sig2 = 0.1; gamlist=logspace(-3,3,60); type='c'; errgamlist=[];
for gam=gamlist,
    [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
    % Obtain classification of test set using trained classifier
    estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha,b}, Xval);
    err = sum(estYval~=Yval); errgamlist = [errgamlist; err];
end

figure('Color',[1 1 1]);
subplot(1,2,1);
semilogx(sig2list, errsiglist./20.*100, 'b-');
title('Test misclassification vs. \sigma (fixed \gamma=1)');
xlabel('\sigma (log_{10})'); ylabel('Misclassification %');

subplot(1,2,2);
semilogx(gamlist, errgamlist./20.*100, 'b-');
title('Test misclassification vs. \gamma (fixed \sigma^2=1)');
xlabel('\gamma (log_{10})'); ylabel('Misclassification %');

export_fig('hyperparams_choice1.pdf')

% Cross validation 10k, and LOOC
perf_10k_CV = crossvalidate({X, Y, 'c', gam, sig2, 'RBF_kernel'}, 10, 'misclass');
perf_LOOCV = leaveoneout({X, Y, 'c', gam, sig2, 'RBF_kernel'}, 'misclass');

% tunelssvm
model_csa = {X, Y, 'c', [], [], 'RBF_kernel', 'csa'};
[gam1, sig21, cost1] = tunelssvm(model_csa, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
[gam2, sig22, cost2] = tunelssvm(model_csa, 'gridsearch', 'crossvalidatelssvm', {10, 'misclass'});
model_ds = {X, Y, 'c', [], [], 'RBF_kernel', 'ds'};
[gam3, sig23, cost3] = tunelssvm(model_ds, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
[gam4, sig24, cost4] = tunelssvm(model_ds, 'gridsearch', 'crossvalidatelssvm', {10, 'misclass'});
