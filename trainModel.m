function trainModel(datasetDir, projectDir, modelName, modelType, numClasses, Hyper_1, Hyper_2, Hyper_3, Hyper_4, Hyper_5, Hyper_6, app, customOutputFcn)
    % Load the dataset
    load(fullfile(datasetDir, 'trainData.mat'), 'trainData_n');
    load(fullfile(datasetDir, 'valData.mat'), 'valData_n');
    load(fullfile(datasetDir, 'testData.mat'), 'testData_n');
    load(fullfile(datasetDir, 'trainLabels.mat'), 'trainLabels');
    load(fullfile(datasetDir, 'valLabels.mat'), 'valLabels');
    load(fullfile(datasetDir, 'testLabels.mat'), 'testLabels');
    valFreq = 30;
    valPat = 9;
    
    switch modelType
        case 0  % CNN Model
            % Reshape data for CNN
            trainData_c = reshape(trainData_n, [size(trainData_n, 1), size(trainData_n, 2), 1, size(trainData_n, 3)]);
            valData_c = reshape(valData_n, [size(valData_n, 1), size(valData_n, 2), 1, size(valData_n, 3)]);
            testData_c = reshape(testData_n, [size(testData_n, 1), size(testData_n, 2), 1, size(testData_n, 3)]);
            
            % Define the CNN network architecture
            sequenceLength = size(trainData_c, 1);
            numFeatures = size(trainData_c, 2);
            numFilters_C1 = Hyper_1;
            numFilters_C2 = Hyper_2;
            dropOut = Hyper_3;
            
            % Training options
            options = trainingOptions("adam", ...
                MaxEpochs=Hyper_4, ...
                Metrics = ["accuracy","fscore"], ...
                InitialLearnRate=Hyper_5, ... 
                MiniBatchSize=Hyper_6, ... 
                Shuffle="every-epoch", ...
                ValidationData={valData_c, valLabels}, ...
                ValidationFrequency=valFreq, ... 
                ValidationPatience=valPat, ...
                OutputNetwork="best-validation", ...
                Verbose=true, ...
                Plots="training-progress");
            
            layers = [
                imageInputLayer([sequenceLength, numFeatures, 1], 'Name', 'input')
                convolution2dLayer([1, 5], numFilters_C1, 'Padding', 'same', 'Name', 'conv_1')
                batchNormalizationLayer('Name', 'batchnorm_1')
                reluLayer('Name', 'relu_1')
                maxPooling2dLayer([1, 2], 'Stride', [1, 2], 'Name', 'maxpool_1')
                convolution2dLayer([1, 3], numFilters_C2, 'Padding', 'same', 'Name', 'conv_2')
                batchNormalizationLayer('Name', 'batchnorm_2')
                reluLayer('Name', 'relu_2')
                maxPooling2dLayer([1, 2], 'Stride', [1, 2], 'Name', 'maxpool_2')
                dropoutLayer(dropOut, 'Name', 'dropout_1')
                fullyConnectedLayer(numClasses, 'Name', 'fc_2')
                softmaxLayer('Name', 'softmax')];
            
            % Train the model
            net = trainnet(trainData_c, trainLabels, layers, "crossentropy", options);
            
            % Test the model
            scores = minibatchpredict(net, testData_c);
            
        case 1  % LSTM Model
            % Reshape data for LSTM
            trainData_l = permute(trainData_n, [2, 1, 3]);
            testData_l = permute(testData_n, [2, 1, 3]);
            valData_l = permute(valData_n, [2, 1, 3]);

            sequenceTrainData = sequence(trainData_l);
            sequenceValData = sequence(valData_l);
            sequenceTestData = sequence(testData_l);

            % Define the LSTM network architecture
            numFeatures = size(trainData_n, 2);
            numHiddenUnits_LSTM = Hyper_1;
            numHiddenUnits_FC = Hyper_2;
            dropOut = Hyper_3;

            % Training options
            options = trainingOptions("adam", ...
                MaxEpochs=Hyper_4, ...
                Metrics = ["accuracy","fscore"], ...
                InitialLearnRate=Hyper_5, ... 
                MiniBatchSize=Hyper_6, ... 
                Shuffle="every-epoch", ...
                ValidationData={sequenceValData, valLabels}, ...
                ValidationFrequency=valFreq, ... 
                ValidationPatience=valPat, ...
                OutputNetwork="best-validation", ...
                Verbose=true, ...
                Plots="training-progress");

            layers = [ ...
                sequenceInputLayer(numFeatures, 'Name', 'input')
                lstmLayer(numHiddenUnits_LSTM, 'OutputMode', 'last', 'Name', 'lstm')
                dropoutLayer(dropOut, 'Name', 'dropout')
                fullyConnectedLayer(numHiddenUnits_FC, 'Name', 'fc_1')
                reluLayer('Name', 'relu')
                fullyConnectedLayer(numClasses, 'Name', 'fc_2')
                softmaxLayer('Name', 'softmax')];

            % Train the model
            net = trainnet(sequenceTrainData, trainLabels, layers, "crossentropy", options);

            % Test the model
            scores = minibatchpredict(net, sequenceTestData);

        case 2  % Transformer Model
            % Reshape data for Transformer
            trainData_t = permute(trainData_n, [2, 1, 3]);
            testData_t = permute(testData_n, [2, 1, 3]);
            valData_t = permute(valData_n, [2, 1, 3]);
            
            sequenceTrainData = sequence(trainData_t);
            sequenceValData = sequence(valData_t);
            sequenceTestData = sequence(testData_t);
            
            % Convert labels to cell array of categorical row vectors with repeated class labels to match the dimensions of sequence data
            numTrainSamples = size(trainLabels, 1);
            trainLabelsCell = cell(numTrainSamples, 1);
            for i = 1:numTrainSamples
                trainLabelsCell{i} = repmat(trainLabels(i, :), size(sequenceTrainData{i}, 1), 1);
            end
            
            numValSamples = size(valLabels, 1);
            valLabelsCell = cell(numValSamples, 1);
            for i = 1:numValSamples
                valLabelsCell{i} = repmat(valLabels(i, :), size(sequenceValData{i}, 1), 1);
            end
            
            % Define the Transformer network architecture
            numFeatures = size(trainData_n, 2);
            numHeads = Hyper_1;
            numHiddenUnits = Hyper_2;
            dropOut = Hyper_3;
            
            % Training options
            options = trainingOptions("adam", ...
                MaxEpochs=Hyper_4, ...
                Metrics = ["accuracy","fscore"], ...
                InitialLearnRate=Hyper_5, ... 
                MiniBatchSize=Hyper_6, ... 
                Shuffle="every-epoch", ...
                ValidationData={sequenceValData, valLabelsCell}, ...
                ValidationFrequency=valFreq, ... 
                ValidationPatience=valPat, ...
                OutputNetwork="best-validation", ...
                Verbose=true, ...
                Plots="training-progress");
            
            layers = [
                sequenceInputLayer(numFeatures, 'Name', 'input')
                selfAttentionLayer(numHeads, numHiddenUnits, 'Name', 'self_attention')
                flattenLayer('Name', 'flatten')
                fullyConnectedLayer(numHiddenUnits, 'Name', 'fc_1')
                reluLayer('Name', 'relu')
                dropoutLayer(dropOut, 'Name', 'dropout')
                fullyConnectedLayer(numClasses, 'Name', 'fc_2')
                softmaxLayer('Name', 'softmax')];
            
            % Train the model
            net = trainnet(sequenceTrainData, trainLabelsCell, layers, "crossentropy", options);
            
            % Test the model
            scores = minibatchpredict(net, sequenceTestData);
            scores = permute(squeeze(scores(1, :, :)), [2, 1]);
    end
    
    predictedLabels = scores2label(scores, unique(trainLabels));
    % Evaluate and display the accuracy
    accuracy = mean(predictedLabels == testLabels);
    fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
    app.TextArea.Value = [app.TextArea.Value; sprintf('Test Accuracy: %.2f%%', accuracy * 100)];

    % Model dir
    fileName = strcat('trained_', modelName, '.mat');
    mat_dir = fullfile(projectDir, "trained_model");
    
    % Ensure the directory exists
    if ~exist(mat_dir, 'dir')
        mkdir(mat_dir);
    end
    
    % Save the trained model to the correct path
    save(fullfile(mat_dir, fileName), 'net');

    % Plot confusion matrix
    figure;
    confusionchart(testLabels, predictedLabels);
    if modelType == 0
        type = 'CNN';
    elseif modelType == 1
        type = 'LSTM';
    else
        type = 'Transformer';
    end
    title(sprintf('Confusion Matrix for Gesture Recognition (Model %s)', type));
end

function sequenceData = sequence(data)
    numSamples = size(data, 3);
    sequenceData = cell(numSamples, 1);
    for i = 1:numSamples
        sequenceData{i} = squeeze(data(:, :, i))';
    end
end
