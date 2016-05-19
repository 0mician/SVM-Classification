load ../dataset/iris
addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

%
% train LS-SVM classifier with linear kernel 
%
type='c'; 
gam = 1; 
disp('Linear kernel'),
[alpha,b] = trainlssvm({X,Y,type,gam,[],'lin_kernel'});
figure('Color',[1 1 1]);
hold on;
subplot(2,2,1);
plotlssvm({X,Y,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);
err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100), 

%
% Train the LS-SVM classifier using polynomial kernel of different degrees
%
type='c'; 
gam = 1; 
t = 1; 
degree = 2;

[alpha,b] = trainlssvm({X,Y,type,gam,[t; degree],'poly_kernel'});
subplot(2,2,2);
plotlssvm({X,Y,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
[Yht, Zt] = simlssvm({X,Y,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)

degree = 3;

[alpha,b] = trainlssvm({X,Y,type,gam,[t; degree],'poly_kernel'});
subplot(2,2,3);
plotlssvm({X,Y,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
[Yht, Zt] = simlssvm({X,Y,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)

degree = 4;

[alpha,b] = trainlssvm({X,Y,type,gam,[t; degree],'poly_kernel'});
subplot(2,2,4);
plotlssvm({X,Y,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
[Yht, Zt] = simlssvm({X,Y,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)

export_fig('iris_linpol.pdf');

%
% use RBF kernel
%

% tune the sig2 while fix gam
%
disp('RBF kernel')
gam = 1; sig2list=[0.01, 0.1, 1, 10];

errlist=[];
i = 1;
for sig2=sig2list,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    subplot(2,2,i);
    plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yt)*100)
    %disp('Press any key to continue...'), pause,
    i = i + 1
end
export_fig('iris_.pdf');

%
% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(log(sig2list), errlist, '*-'), 
xlabel('log(sig2)'), ylabel('number of misclass'),

% Overfitting in sigma and high polynomials degree

gam = 1; sig2list=[0.001, 0.01]

errlist=[];
i = 1;
for sig2=sig2list,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    subplot(2,2,i);
    plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yt)*100)
    %disp('Press any key to continue...'), pause,
    i = i + 1
end

type='c'; 
gam = 1; 
t = 1; 
degree = 15;

[alpha,b] = trainlssvm({X,Y,type,gam,[t; degree],'poly_kernel'});
subplot(2,2,3);
plotlssvm({X,Y,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
[Yht, Zt] = simlssvm({X,Y,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)

degree = 20;

[alpha,b] = trainlssvm({X,Y,type,gam,[t; degree],'poly_kernel'});
subplot(2,2,4);
plotlssvm({X,Y,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
[Yht, Zt] = simlssvm({X,Y,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)
export_fig('iris_overfitting.pdf');