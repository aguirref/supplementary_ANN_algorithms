function [A]=DB_rescale_ANN_train(resol,limit_output,database,varargin)


    %% This part of the code just performs minor integrity checks of the arguemnts passed to the function
    save_pref=1;
    if mod(nargin-2,2)
        display('A odd number of arguments have been introduced. Please revise the function call');
    else
        for i=1:nargin-2
            if strcmp(varargin{i},'save?')
                save_pref=varargin{i+1};
            end
        end
    end
    if limit_output==1
        lims='limited_range';
    else
        lims='unlimited_range';
    end

    training_routine='old';
    
    %% This part of the code downloads the MNIST dataset (train and test)  
    % This first part downloads the files
    if ~exist(fullfile('..','..','..','DBs',database),'dir')                %This line checks whether the destination folder exist or not and creates it if necesary.
        mkdir(fullfile('..','..','..','DBs',database));
    end    
    if ~exist(fullfile('..','..','..','DBs',database,'train-images.idx3-ubyte.gz'),'file')
        websave(fullfile('..','..','..','DBs',database,'train-images.idx3-ubyte.gz'),'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz');
        
    end
    if ~exist(fullfile('..','..','..','DBs',database,'train-labels.idx1-ubyte.gz'),'file')
        websave(fullfile('..','..','..','DBs',database,'train-labels.idx1-ubyte.gz'),'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz');
    end
    if ~exist(fullfile('..','..','..','DBs',database,'t10k-images.idx3-ubyte.gz'),'file')
        websave(fullfile('..','..','..','DBs',database,'t10k-images.idx3-ubyte.gz'),'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz');
    end
    if ~exist(fullfile('..','..','..','DBs',database,'t10k-labels.idx1-ubyte.gz'),'file')
        websave(fullfile('..','..','..','DBs',database,'t10k-labels.idx1-ubyte.gz'),'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz');
    end
    
    % and this one unzip them
    if ~exist(fullfile('..','..','..','DBs',database,'train-images.idx3-ubyte'),'file')
        gunzip(fullfile('..','..','..','DBs',database,'train-images.idx3-ubyte.gz'));
    end
    if ~exist(fullfile('..','..','..','DBs',database,'train-labels.idx1-ubyte'),'file')
        gunzip(fullfile('..','..','..','DBs',database,'train-labels.idx1-ubyte.gz'));
    end
    if ~exist(fullfile('..','..','..','DBs',database,'t10k-images.idx3-ubyte'),'file')
        gunzip(fullfile('..','..','..','DBs',database,'t10k-images.idx3-ubyte.gz'));
    end
    if ~exist(fullfile('..','..','..','DBs',database,'t10k-labels.idx1-ubyte'),'file')
        gunzip(fullfile('..','..','..','DBs',database,'t10k-labels.idx1-ubyte.gz'));
    end    
    
    %% This part of the code loads the downloadad database in memory. It also contemplates the case of using the CIFAR-10 dataset, although it is not suitable to test SLPs
    if strcmpi(database,'MNIST')
        ref_resol=[28 28];
        
        images = loadMNISTImages(fullfile('..','..','..','DBs',database,'train-images.idx3-ubyte'));
        labels = loadMNISTLabels(fullfile('..','..','..','DBs',database,'train-labels.idx1-ubyte'))';

        images_t10k = loadMNISTImages(fullfile('..','..','..','DBs',database,'t10k-images.idx3-ubyte'));
        labels_t10k = loadMNISTLabels(fullfile('..','..','..','DBs',database,'t10k-labels.idx1-ubyte'))'; 
        
    elseif strcmpi(database,'CIFAR-10')
        ref_resol=[32 32];

        images = load(fullfile('..','..','..','DBs',database,'train-images-idx3-ubyte'),'data_grayscale');
        images = images.data_grayscale;
        labels = load(fullfile('..','..','..','DBs',database,'train-labels-idx1-ubyte'),'data_labels')';
        labels = double(labels.data_labels);

        images_t10k = load(fullfile('..','..','..','DBs',database,'t10k-images-idx3-ubyte'),'data_test_grayscale');
        images_t10k = images_t10k.data_test_grayscale;
        labels_t10k = load(fullfile('..','..','..','DBs',database,'t10k-labels-idx1-ubyte'),'data_test_labels')';  
        labels_t10k = double(labels_t10k.data_test_labels);
        
    end
    
    %% this part of the code performs the dataset re-scaling
    % this block re-scales the train dataset
    for i = 1:size(images,2)                                        	
        digit = reshape(images(:, i), [ref_resol(1), ref_resol(2)]);                     
        digit_resized = imresize(digit,[resol(1) resol(2)]);
        if limit_output==1
            digit_resized(digit_resized>1)=1;
            digit_resized(digit_resized<0)=0;
        end
        eval(sprintf('images_%dx%d(:,i) = reshape(digit_resized, [resol(1)*resol(2),1]);',resol(1),resol(2)));
    end
    eval(sprintf('labels_%dx%d = labels;',resol(1),resol(2)));

    % and this part re-scales de test dataset.
    for i = 1:size(images_t10k,2)                                        
        digit = reshape(images_t10k(:, i), [ref_resol(1), ref_resol(2)]);                   
        digit_resized = imresize(digit,[resol(1) resol(2)]);
        if limit_output==1
            digit_resized(digit_resized>1)=1;
            digit_resized(digit_resized<0)=0;
        end
        eval(sprintf('images_t10k_%dx%d(:,i) = reshape(digit_resized, [resol(1)*resol(2),1]);',resol(1),resol(2)));
    end
    eval(sprintf('labels_t10k_%dx%d = labels_t10k;',resol(1),resol(2)));

    %This part checks if the destination folder for the re-scaled database
    %exist. if not, it creates it
    if ~exist(fullfile('..','..','..','DBs',database, lims, sprintf('%dby%d',resol(1), resol(2))),'dir')
        mkdir(fullfile('..','..','..','DBs',database, lims, sprintf('%dby%d',resol(1), resol(2))));
    end
    
    %This part saves the re-scaled dataset to the previoulsy creacted
    %folder
    if save_pref == 1
        save(fullfile('..','..','..','DBs',database,lims,sprintf('%dby%d',resol(1), resol(2)),sprintf('train-images_%dx%d.mat',resol(1),resol(2))),...
                                                                                        sprintf('images_%dx%d',resol(1),resol(2)),...
                                                                                        sprintf('labels_%dx%d',resol(1),resol(2)),...
                                                                                        sprintf('images_t10k_%dx%d',resol(1),resol(2)),...
                                                                                        sprintf('labels_t10k_%dx%d',resol(1),resol(2)));
    end
    
    %% This part shows the first 10 images of the dataset, in the original and re-scaled sizes.  
    figure();
    colormap(gray);
    for i=1:10
        subplot(2,10,i)
        digit = reshape(images(:, i), [ref_resol(1), ref_resol(2)]);
        imagesc(digit);
        axis square;
    end
    
    for i=1:10
        subplot(2,10,i+10)
        eval(sprintf('digit = images_%dx%d(:,i);',resol(1),resol(2)));
        digit = reshape(digit, [resol(1), resol(2)]);
        imagesc(digit);
        axis square;
    end
    
    eval(sprintf('x = images_%dx%d;',resol(1),resol(2)));
    eval(sprintf('t = labels_%dx%d;',resol(1),resol(2)));
    eval(sprintf('t_10k = labels_t10k_%dx%d;',resol(1),resol(2)));
    
    %% This part creates and trains the neural network.
    if strcmpi(training_routine,'old')

        trainFcn = 'trainscg';                                                  % use scaled conjugate gradient for training

        for j=1:10
            for i=1:20
                
                A=patternnet(10,'trainscg');                                            % Network creation
                A.inputs{1,1}.size=resol(1)*resol(2);
                A.layers{2,1}.size=10;
                A.LW{2,1}=diag(ones(10,1));
                A.layerweights{2,1}.learn=0;
                A.inputweights{1,1}.learnFcn='learncon';
                A.layers{1,1}.transferFcn='purelin';
                A.biases{1,1}.learn=0;
                A.biases{2,1}.learn=0;
                A.biasConnect=[0;0];

                A.divideFcn='dividerand';
                A.divideMode='sample';
                A.divideParam.trainRatio=80/100;
                A.divideParam.valRatio=20/100;
                A.divideParam.testRatio=0/100;

                A.trainParam.max_fail=15;
                A.trainParam.min_grad=1e-7;
                t_vec=full(ind2vec(t+1,10));
                
                A.trainParam.epochs=i;
                [A,tr_a] = train(A,x,t_vec);                                            %Network Training 

                %This parts extract the synaptic weights from the net object A, and
                %stores them to differnt variables.
                G_real=A.IW{1,1};
                G=G_real;
                G(G<0)=0;

                G_enf=G;
                G_enf=log(G_enf+1)/log(2);

                G_pos=G_real;
                G_pos(G_pos<0)=0;

                G_neg=G_real;
                G_neg(G_neg>0)=0;

                G_neg=G_neg*-1;

                G_bin=G_pos;
                G_bin(G_bin>0)=1;

                G_bin2=G_pos;
                G_bin2(G_bin2>0.5)=1;
                G_bin2(G_bin2<0.5)=0;

                %this part of the tests the trained network. Look for the "Accuracy" variable in the workspace shown in the right panel. it sholuld be around 90% 
                eval(sprintf('results=A(images_t10k_%dx%d);',resol(1),resol(2)));
                [~,results]=max(results);
                results=results-1;
                results_test=results;
                eval(sprintf('comparison=(results==labels_t10k_%dx%d);',resol(1),resol(2)));
                Accuracy=sum(comparison)/length(comparison)*100;
                accuracy_test(i,j)=Accuracy;

                eval(sprintf('results_train=A(images_%dx%d);',resol(1),resol(2)));
                [~,results_train]=max(results_train);
                results_train=results_train-1;
                eval(sprintf('comparison_train=(results_train==labels_%dx%d);',resol(1),resol(2)));
                accuracy_train(i,j)=sum(comparison_train)/length(comparison_train)*100;

                for k=0:(10-1)
                    DIGITS{k+1,1,i,j}=k;
                    DIGITS{k+1,2,i,j}=find(t_10k(1,1:10000)==k);

                    ERRORS{k+1,1,i,j}=k;
                    ERRORS{k+1,2,i,j}=find(results_test~=t_10k & t_10k==k);
                    ERRORS{k+1,3,i,j}=sum(results_test~=t_10k & t_10k==k)/size(DIGITS{k+1,2},2);

                    TP{k+1,1,i,j}=k;
                    TP{k+1,2,i,j}=find(results_test==k & t_10k==k);
                    TP{k+1,3,i,j}=sum(results_test==k & t_10k==k);

                    TN{k+1,1,i,j}=k;
                    TN{k+1,2,i,j}=find(results_test~=k & t_10k~=k);
                    TN{k+1,3,i,j}=sum(results_test~=k & t_10k~=k);

                    FP{k+1,1,i,j}=k;
                    FP{k+1,2,i,j}=find(results_test==k & t_10k~=k);
                    FP{k+1,3,i,j}=sum(results_test==k & t_10k~=k);

                    FN{k+1,1,i,j}=k;
                    FN{k+1,2,i,j}=find(results_test~=k & t_10k==k);
                    FN{k+1,3,i,j}=sum(results_test~=k & t_10k==k);                    

                    accuracy_I(k+1,i,j) = (TP{k+1,3,i,j})/length(find(results_test==k));
                    accuracy_II(k+1,i,j) = (TP{k+1,3,i,j}+TN{k+1,3,i,j})/(TP{k+1,3,i,j}+TN{k+1,3,i,j}+FP{k+1,3,i,j}+FN{k+1,3,i,j});
                    precision(k+1,i,j) =TP{k+1,3,i,j} / (TP{k+1,3,i,j} + FP{k+1,3,i,j});
                    sensitivity(k+1,i,j) = TP{k+1,3,i,j} / (TP{k+1,3,i,j} + FN{k+1,3,i,j}); 
                    specificity(k+1,i,j) = TN{k+1,3,i,j} / (FP{k+1,3,i,j} + TN{k+1,3,i,j}); 
                    F1_score(k+1,i,j) = 2*TP{k+1,3,i,j} /(2*TP{k+1,3,i,j} + FP{k+1,3,i,j} + FN{k+1,3,i,j});
                                
                end                
                
                FN_tot(i,j)=0;
                FP_tot(i,j)=0;
                TP_tot(i,j)=0;
                TN_tot(i,j)=0;
                for index_class=1:10  
                    TP_tot(i,j)=TP_tot(i,j)+TP{index_class,3,i,j};
                    TN_tot(i,j)=TN_tot(i,j)+TN{index_class,3,i,j};
                    FP_tot(i,j)=FP_tot(i,j)+FP{index_class,3,i,j};
                    FN_tot(i,j)=FN_tot(i,j)+FN{index_class,3,i,j};
                end

                PRECISION_gral(i,j) = TP_tot(i,j)/(FP_tot(i,j)+TP_tot(i,j));
                SENSITIVITY_gral(i,j) = TP_tot(i,j)/(FN_tot(i,j)+TP_tot(i,j));
                SPECIFICITY_gral(i,j) = TN_tot(i,j)/(FP_tot(i,j)+TN_tot(i,j));
                F1_score_gral(i,j) = 2*TP_tot(i,j)/(2*TP_tot(i,j) + FP_tot(i,j) + FN_tot(i,j));
                kappa_score_gral(i,j) = (accuracy_test(i,j)./100-0.1)./(1-0.1);

                if j==1 && i==1
                    h=figure();
                    h1=subplot(2,4,1);
                    %plot(1:1:i,nonzeros(accuracy_train(:,j)),'Marker','o','Color','r','MarkerSize',5.0);
                    %hold on
                    plot(1:1:i,nonzeros(accuracy_test(:,j)),'Marker','o','Color','b','MarkerSize',5.0);
                    hold on
                    xlabel('Epochs');
                    ylabel('Accuracy [%]');
                    
                    h2=subplot(2,4,4);
%                     plot(1:1:i,nonzeros(SENSITIVITY_gral(:,j))*100,'Marker','o','Color','r','MarkerSize',5.0);
%                     hold on
                    for z = 1:3
                        plot(1:1:i,nonzeros(sensitivity(z,:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
                        hold on         
                    end
                    xlabel('Epochs');
                    ylabel('Sensitivity [%]');
                    
                    h3=subplot(2,4,5);
%                     plot(1:1:i,nonzeros(SPECIFICITY_gral(:,j))*100,'Marker','o','Color','r','MarkerSize',5.0);
%                     hold on
                    for z = 1:3
                        plot(1:1:i,nonzeros(specificity(z,:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
                        hold on         
                    end
                    xlabel('Epochs');
                    ylabel('Specificity [%]');
                    
                    h4=subplot(2,4,6);
%                     plot(1:1:i,nonzeros(PRECISION_gral(:,j))*100,'Marker','o','Color','r','MarkerSize',5.0);
%                     hold on
                    for z = 1:3
                        plot(1:1:i,nonzeros(precision(z,:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
                        hold on         
                    end
                    xlabel('Epochs');
                    ylabel('Precision [%]');
                    
                    h5=subplot(2,4,7);
%                     plot(1:1:i,nonzeros(F1_score_gral(:,j))*100,'Marker','o','Color','r','MarkerSize',5.0);
%                     hold on
                    for z = 1:3
                        plot(1:1:i,nonzeros(F1_score(z,:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
                        hold on         
                    end                        
                    xlabel('Epochs');
                    ylabel('F1 score [%]');
                    
                    h6=subplot(2,4,8);
                    plot(1:1:i,nonzeros(kappa_score_gral(:,j)),'Marker','o','Color','r','MarkerSize',5.0);
                    hold on
                    xlabel('Epochs');
                    ylabel('\kappa-coefficient [a.u.]');                    
                else
                    axes(h1);
                    axis([0 i+1, 50 100]);
%                     plot(1:1:i,nonzeros(accuracy_train(:,j)),'Marker','o','Color','r','MarkerSize',5.0);
%                     hold on
                    plot(1:1:i,nonzeros(accuracy_test(:,j)),'Marker','o','Color','b','MarkerSize',5.0);

                    axes(h2);
                    axis([0 i+1, 50 100]);
%                     plot(1:1:i,nonzeros(SENSITIVITY_gral(:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
%                     hold on         
                    for z = 1:3
                        plot(1:1:i,nonzeros(sensitivity(z,:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
                        hold on         
                    end

                    axes(h3);
                    axis([0 i+1, 50 100]);
%                     plot(1:1:i,nonzeros(SPECIFICITY_gral(:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
%                     hold on                                  
                    for z = 1:3
                        plot(1:1:i,nonzeros(specificity(z,:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
                        hold on         
                    end

                    axes(h4);
                    axis([0 i+1, 50 100]);
%                     plot(1:1:i,nonzeros(PRECISION_gral(:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
%                     hold on            
                    for z = 1:3
                        plot(1:1:i,nonzeros(precision(z,:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
                        hold on         
                    end

                    axes(h5);
                    axis([0 i+1, 50 100]);
%                     plot(1:1:i,nonzeros(F1_score_gral(:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
%                     hold on   
                    for z = 1:3
                        plot(1:1:i,nonzeros(F1_score(z,:,j))*100,'Marker','o','Color','b','MarkerSize',5.0);
                        hold on         
                    end    

                    axes(h6);
                    axis([0 i+1, 0 1]);
                    plot(1:1:i,nonzeros(kappa_score_gral(:,j)),'Marker','o','Color','b','MarkerSize',5.0);
                    hold on                       
                end                                  
            end
            h7=subplot(2,4,2);
            plot(tr_a.epoch,nonzeros(tr_a.perf),'Marker','o','Color','r','MarkerSize',5.0);
            hold on
            xlabel('Epochs');
            ylabel('Loss(Cross-entropy)');
            
            if ~exist('loss', 'var')
                loss = [tr_a.epoch', tr_a.perf'];
            else
                loss = [loss, tr_a.perf'];
            end

            h8=subplot(2,4,3);
            edges=-0.5:1:(10-0.5);
            for digit=0:(10-1)
                PROB(digit+1,:)=histcounts(results_test(1,t_10k==digit),edges)/size(DIGITS{digit+1,2},2);
                %CURR_MAT(digit+1,:)=CURRENTS{digit+1,3};
                %PROB_MAT(digit+1,:)=CURRENTS{digit+1,4};
            end
            imagesc(PROB);
            colorbar

            zlabel('Digit probability per neuron');
            ylabel('Digit [#]');
            yticks([1:1:10]);
            yticklabels(cellstr(split(num2str([0:1:10-1]))));
            %yticklabels({'0','1','2','3','4','5','6','7','8','9'});
            xlabel({'Output','Neuron [#]'});
            xticks([1:1:10]);
            xticklabels(cellstr(split(num2str([0:1:10-1]))));
            %xticklabels({'0','1','2','3','4','5','6','7','8','9'});
            axis square
            axis([0 10+1 0 10+1]);
            
        end
        
        
    else
        
        layers = [
        imageInputLayer([resol(1)*resol(2) 1 1])
        fullyConnectedLayer(10,'BiasLearnRateFactor',0,'Bias',zeros(10,1))
        softmaxLayer()
        classificationLayer];
        
        opts = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 8, ...
        'L2Regularization', 0.004, ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 100, ...
        'Verbose', true);

        [net, info] = trainNetwork(x, t, layers, opts)
    end

    save('variables_for_python.mat','loss', 'accuracy_test', 'specificity', 'sensitivity', 'precision','F1_score', 'accuracy_I', 'PROB')

end

