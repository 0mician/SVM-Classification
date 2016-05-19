load ../dataset/iris
addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

gam = 1; type='c'; sig2list=logspace(-3,3,60); errlist=[];

for sig2=sig2list,
    [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
    % Obtain classification of test set using trained classifier
    [Yht, Zt] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); errlist = [errlist; err]; 
end

figure('Color',[1 1 1]);
semilogx(sig2list, errlist./20.*100, 'b-');
title('Test missclassification vs. Sigma');
xlabel('Sigma (log_{10})'); ylabel('Missclassification %');

export_fig('rbf_sigma.pdf')