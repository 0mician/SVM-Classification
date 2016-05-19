load ../dataset/breast
addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

boxplot(trainset,'orientation','horizontal');

% data visualization with PCA
[coeff,scores,~,~,explained] = pca(trainset,'VariableWeights','variance');
figure('Color',[1 1 1]);
subplot(1,2,1);
scatter3(scores(:,1), scores(:,2), scores(:,3), 15, labels_train);
title('Graph of the first 3 PC'); xlabel('PC 1'); ylabel('PC 2'); zlabel('PC3');
subplot(2,2,2);
gscatter(scores(:,1), scores(:,2), labels_train, 'br', 'ox');
title('Graph of the first 2 PC'); xlabel('PC 1'); ylabel('PC 2');
subplot(2,2,4);
bar(explained(1:5));
title('Principal components importance'); xlabel('PC number'); ylabel('Variability explained (%)');

export_fig('breast_pcviz.pdf')

% Linear model
gamlist=logspace(-3,3); type='c'; errlist=[];

for gam=gamlist,
    [alpha,b] = trainlssvm({trainset,labels_train,type,gam,[],'lin_kernel'});
    [Yht, Ylin] = simlssvm({trainset,labels_train,type,gam,[],'lin_kernel'}, {alpha,b}, testset);
    err = sum(Yht~=labels_test); errlist = [errlist; err];
end
[min, index] = min(errlist);
[Yht, Ylin] = simlssvm({trainset,labels_train,type,gamlist(index),[],'lin_kernel'}, {alpha,b}, testset);
perf_lin = (1-errlist(index)/numel(labels_test))*100;

% RBF with tunning 
model_csa = {trainset, labels_train, 'c', [], [], 'RBF_kernel', 'csa'};
[gam, sig2, cost] = tunelssvm(model_csa, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});

[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'});
[estYval, YRBF] = simlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'}, {alpha,b}, testset);
err = sum(estYval~=labels_test); perf_rbf = (1-err/numel(labels_test))*100;

% ROC curve
figure('Color',[1 1 1]);
roc(Ylin, labels_test);
hold on;
roc(YRBF, labels_test);

export_fig('breast_ROC.pdf')
