load ../dataset/iris
addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

sig2 = 0.1; gamlist=logspace(-3,3,60); type='c'; errlist=[];

for gam=gamlist,
    [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
    % Obtain classification of test set using trained classifier
    [Yht, Zt] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); errlist = [errlist; err];
end

figure('Color',[1 1 1]);
semilogx(gamlist, errlist./20.*100, 'b-');
title('Test missclassification vs. Gamma (fixed \sigma^2=0.1)');
xlabel('\gamma (log_{10})'); ylabel('Missclassification %');

export_fig('rbf_gamma.pdf')