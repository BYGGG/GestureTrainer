function trainModel(datasetDir, projectDir, modelName, modelType, numClasses, Hyper_1, Hyper_2, Hyper_3, Hyper_4, Hyper_5, Hyper_6, app, customOutputFcn)
    % Load the dataset
    load(fullfile(datasetDir, 'trainData.mat'), 'trainData_n');
    load(fullfile(datasetDir, 'valData.mat'), 'valData_n');
    load(fullfile(datasetDir, 'testData.mat'), 'testData_n');
    load(fullfile(datasetDir, 'trainLabels.mat'), 'trainLabels');
    load(fullfile(datasetDir, 'valLabels.mat'), 'valLabels');
    load(fullfile(datasetDir, 'testLabels.mat'), 'testLabels');

    % Load traceability info if available
    hasTraceability = false;
    if isfile(fullfile(datasetDir, 'testParticipants.mat'))
        load(fullfile(datasetDir, 'testParticipants.mat'), 'testParticipants');
        load(fullfile(datasetDir, 'testFilenames.mat'), 'testFilenames');
        hasTraceability = true;
    end

    % Load dataset customisation if available
    datasetConfig = struct();
    jsonPath = fullfile(datasetDir, 'dataset_customisation.json');
    if isfile(jsonPath)
        fid = fopen(jsonPath, 'r');
        rawJson = fread(fid, '*char')';
        fclose(fid);
        datasetConfig = jsondecode(rawJson);
    end

    valFreq = 30;
    valPat = 9;

    % =====================================================================
    % Create analysis output directory
    % =====================================================================
    timestamp = datestr(now, 'yyyymmdd_HHMMss');
    if modelType == 0
        typeStr = 'CNN';
    elseif modelType == 1
        typeStr = 'LSTM';
    else
        typeStr = 'Transformer';
    end
    analysisDir = fullfile(projectDir, 'analysis', sprintf('%s_%s_%s', modelName, typeStr, timestamp));
    if ~exist(analysisDir, 'dir')
        mkdir(analysisDir);
    end
    fprintf('Analysis output directory: %s\n', analysisDir);

    % =====================================================================
    % Build and train model
    % =====================================================================
    switch modelType
        case 0  % CNN Model
            trainData_c = reshape(trainData_n, [size(trainData_n, 1), size(trainData_n, 2), 1, size(trainData_n, 3)]);
            valData_c = reshape(valData_n, [size(valData_n, 1), size(valData_n, 2), 1, size(valData_n, 3)]);
            testData_c = reshape(testData_n, [size(testData_n, 1), size(testData_n, 2), 1, size(testData_n, 3)]);

            sequenceLength = size(trainData_c, 1);
            numFeatures = size(trainData_c, 2);
            numFilters_C1 = Hyper_1;
            numFilters_C2 = Hyper_2;
            dropOut = Hyper_3;

            options = trainingOptions("adam", ...
                MaxEpochs=Hyper_4, ...
                Metrics=["accuracy","fscore"], ...
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

            net = trainnet(trainData_c, trainLabels, layers, "crossentropy", options);
            scores = minibatchpredict(net, testData_c);

        case 1  % LSTM Model
            trainData_l = permute(trainData_n, [2, 1, 3]);
            testData_l = permute(testData_n, [2, 1, 3]);
            valData_l = permute(valData_n, [2, 1, 3]);

            sequenceTrainData = sequence(trainData_l);
            sequenceValData = sequence(valData_l);
            sequenceTestData = sequence(testData_l);

            numFeatures = size(trainData_n, 2);
            numHiddenUnits_LSTM = Hyper_1;
            numHiddenUnits_FC = Hyper_2;
            dropOut = Hyper_3;

            options = trainingOptions("adam", ...
                MaxEpochs=Hyper_4, ...
                Metrics=["accuracy","fscore"], ...
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

            net = trainnet(sequenceTrainData, trainLabels, layers, "crossentropy", options);
            scores = minibatchpredict(net, sequenceTestData);

        case 2  % Transformer Model
            trainData_t = permute(trainData_n, [2, 1, 3]);
            testData_t = permute(testData_n, [2, 1, 3]);
            valData_t = permute(valData_n, [2, 1, 3]);

            sequenceTrainData = sequence(trainData_t);
            sequenceValData = sequence(valData_t);
            sequenceTestData = sequence(testData_t);

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

            numFeatures = size(trainData_n, 2);
            numHeads = Hyper_1;
            numHiddenUnits = Hyper_2;
            dropOut = Hyper_3;

            options = trainingOptions("adam", ...
                MaxEpochs=Hyper_4, ...
                Metrics=["accuracy","fscore"], ...
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

            net = trainnet(sequenceTrainData, trainLabelsCell, layers, "crossentropy", options);
            scores = minibatchpredict(net, sequenceTestData);
            scores = permute(squeeze(scores(1, :, :)), [2, 1]);
    end

    % =====================================================================
    % Compute predictions and overall accuracy
    % =====================================================================
    classNames = unique(trainLabels);
    predictedLabels = scores2label(scores, classNames);
    accuracy = mean(predictedLabels == testLabels);
    fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
    app.TextArea.Value = [app.TextArea.Value; sprintf('Test Accuracy: %.2f%%', accuracy * 100)];

    % =====================================================================
    % Save trained model
    % =====================================================================
    fileName = strcat('trained_', modelName, '.mat');
    mat_dir = fullfile(projectDir, "trained_model");
    if ~exist(mat_dir, 'dir')
        mkdir(mat_dir);
    end
    save(fullfile(mat_dir, fileName), 'net');

    % =====================================================================
    % POST-TRAINING ANALYSIS
    % =====================================================================
    fprintf('\nGenerating post-training analysis...\n');

    % Get class names as cell array of strings
    classNamesList = cellstr(classNames);
    numClassesActual = length(classNamesList);

    % Build confusion matrix manually for full metric computation
    testLabelsNum = grp2idx(testLabels);
    predLabelsNum = grp2idx(predictedLabels);
    cm = zeros(numClassesActual, numClassesActual);
    for i = 1:length(testLabelsNum)
        cm(testLabelsNum(i), predLabelsNum(i)) = cm(testLabelsNum(i), predLabelsNum(i)) + 1;
    end

    % -----------------------------------------------------------------
    % 1. Confusion Matrix (Counts)
    % -----------------------------------------------------------------
    fig1 = figure('Visible', 'off', 'Position', [100 100 800 650]);
    confusionchart(testLabels, predictedLabels, ...
        'Title', sprintf('Confusion Matrix — %s (%s)', modelName, typeStr), ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    exportgraphics(fig1, fullfile(analysisDir, 'confusion_matrix.png'), 'Resolution', 300);
    savefig(fig1, fullfile(analysisDir, 'confusion_matrix.fig'));
    close(fig1);
    fprintf('  Saved: confusion_matrix.png\n');

    % -----------------------------------------------------------------
    % 2. Confusion Matrix (Normalised — percentages)
    % -----------------------------------------------------------------
    cm_norm = cm ./ (sum(cm, 2) + 1e-10);

    fig2 = figure('Visible', 'off', 'Position', [100 100 800 650]);
    heatmap(classNamesList, classNamesList, round(cm_norm, 3), ...
        'Title', sprintf('Normalised Confusion Matrix — %s (%s)', modelName, typeStr), ...
        'XLabel', 'Predicted Label', ...
        'YLabel', 'True Label', ...
        'ColorbarVisible', 'on', ...
        'Colormap', parula, ...
        'CellLabelFormat', '%.2f');
    exportgraphics(fig2, fullfile(analysisDir, 'confusion_matrix_normalised.png'), 'Resolution', 300);
    savefig(fig2, fullfile(analysisDir, 'confusion_matrix_normalised.fig'));
    close(fig2);
    fprintf('  Saved: confusion_matrix_normalised.png\n');

    % -----------------------------------------------------------------
    % 3. Compute per-class metrics (Precision, Recall, F1)
    % -----------------------------------------------------------------
    precision = zeros(numClassesActual, 1);
    recall = zeros(numClassesActual, 1);
    f1 = zeros(numClassesActual, 1);
    support = sum(cm, 2);

    for c = 1:numClassesActual
        tp = cm(c, c);
        fp = sum(cm(:, c)) - tp;
        fn = sum(cm(c, :)) - tp;

        if (tp + fp) > 0
            precision(c) = tp / (tp + fp);
        end
        if (tp + fn) > 0
            recall(c) = tp / (tp + fn);
        end
        if (precision(c) + recall(c)) > 0
            f1(c) = 2 * precision(c) * recall(c) / (precision(c) + recall(c));
        end
    end

    macroPrecision = mean(precision);
    macroRecall = mean(recall);
    macroF1 = mean(f1);

    weights = support / (sum(support) + 1e-10);
    weightedPrecision = sum(precision .* weights);
    weightedRecall = sum(recall .* weights);
    weightedF1 = sum(f1 .* weights);

    % -----------------------------------------------------------------
    % 4. Per-Class Accuracy Bar Chart
    % -----------------------------------------------------------------
    fig3 = figure('Visible', 'off', 'Position', [100 100 max(900, numClassesActual*60) 500]);
    x = 1:numClassesActual;

    barColors = [0.3 0.5 0.8] .* ones(numClassesActual, 1) + recall .* [0.3 0.2 0.0];
    b = bar(x, recall, 'FaceColor', 'flat');
    b.CData = barColors;
    hold on;
    yline(accuracy, 'r--', 'LineWidth', 2, 'Label', sprintf('Overall: %.3f', accuracy), ...
        'LabelHorizontalAlignment', 'left');

    for i = 1:numClassesActual
        text(i, recall(i) + 0.02, sprintf('%.2f\n(%d)', recall(i), support(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end

    set(gca, 'XTick', x, 'XTickLabel', classNamesList, 'XTickLabelRotation', 45);
    xlabel('Gesture Class', 'FontWeight', 'bold');
    ylabel('Accuracy (Recall)', 'FontWeight', 'bold');
    title(sprintf('Per-Class Accuracy — %s (%s)', modelName, typeStr), 'FontWeight', 'bold');
    ylim([0, 1.15]);
    grid on;
    set(gca, 'GridAlpha', 0.3);
    hold off;

    exportgraphics(fig3, fullfile(analysisDir, 'per_class_accuracy.png'), 'Resolution', 300);
    savefig(fig3, fullfile(analysisDir, 'per_class_accuracy.fig'));
    close(fig3);
    fprintf('  Saved: per_class_accuracy.png\n');

    % -----------------------------------------------------------------
    % 5. Precision / Recall / F1 Grouped Bar Chart
    % -----------------------------------------------------------------
    fig4 = figure('Visible', 'off', 'Position', [100 100 max(900, numClassesActual*60) 500]);
    metricsMatrix = [precision, recall, f1];
    b = bar(x, metricsMatrix, 'grouped');
    b(1).FaceColor = [0.2 0.4 0.8];
    b(2).FaceColor = [0.8 0.3 0.2];
    b(3).FaceColor = [0.2 0.7 0.3];
    legend('Precision', 'Recall', 'F1-Score', 'Location', 'southoutside', 'Orientation', 'horizontal');

    set(gca, 'XTick', x, 'XTickLabel', classNamesList, 'XTickLabelRotation', 45);
    xlabel('Gesture Class', 'FontWeight', 'bold');
    ylabel('Score', 'FontWeight', 'bold');
    title(sprintf('Classification Metrics — %s (%s)', modelName, typeStr), 'FontWeight', 'bold');
    ylim([0, 1.15]);
    grid on;
    set(gca, 'GridAlpha', 0.3);

    exportgraphics(fig4, fullfile(analysisDir, 'classification_metrics.png'), 'Resolution', 300);
    savefig(fig4, fullfile(analysisDir, 'classification_metrics.fig'));
    close(fig4);
    fprintf('  Saved: classification_metrics.png\n');

    % -----------------------------------------------------------------
    % 6. Classification Report (text file)
    % -----------------------------------------------------------------
    reportPath = fullfile(analysisDir, 'classification_report.txt');
    fid = fopen(reportPath, 'w');
    fprintf(fid, 'Classification Report\n');
    fprintf(fid, '=====================================================================\n');
    fprintf(fid, '%-25s %10s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score', 'Support');
    fprintf(fid, '---------------------------------------------------------------------\n');
    for c = 1:numClassesActual
        fprintf(fid, '%-25s %10.3f %10.3f %10.3f %10d\n', ...
            classNamesList{c}, precision(c), recall(c), f1(c), support(c));
    end
    fprintf(fid, '---------------------------------------------------------------------\n');
    fprintf(fid, '%-25s %10.3f %10.3f %10.3f %10d\n', ...
        'Macro Avg', macroPrecision, macroRecall, macroF1, sum(support));
    fprintf(fid, '%-25s %10.3f %10.3f %10.3f %10d\n', ...
        'Weighted Avg', weightedPrecision, weightedRecall, weightedF1, sum(support));
    fprintf(fid, '=====================================================================\n');
    fprintf(fid, '\nOverall Accuracy: %.4f\n', accuracy);
    fclose(fid);
    fprintf('  Saved: classification_report.txt\n');

    % -----------------------------------------------------------------
    % 7. LaTeX Table
    % -----------------------------------------------------------------
    latexPath = fullfile(analysisDir, 'classification_table.tex');
    fid = fopen(latexPath, 'w');
    fprintf(fid, '\\begin{table}[htbp]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\caption{Classification Results — %s (%s)}\n', modelName, typeStr);
    fprintf(fid, '\\begin{tabular}{lcccc}\n');
    fprintf(fid, '\\toprule\n');
    fprintf(fid, '\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Support} \\\\\n');
    fprintf(fid, '\\midrule\n');
    for c = 1:numClassesActual
        clsLatex = strrep(classNamesList{c}, '_', '\_');
        fprintf(fid, '%s & %.3f & %.3f & %.3f & %d \\\\\n', ...
            clsLatex, precision(c), recall(c), f1(c), support(c));
    end
    fprintf(fid, '\\midrule\n');
    fprintf(fid, 'Macro Avg & %.3f & %.3f & %.3f & %d \\\\\n', ...
        macroPrecision, macroRecall, macroF1, sum(support));
    fprintf(fid, 'Weighted Avg & %.3f & %.3f & %.3f & %d \\\\\n', ...
        weightedPrecision, weightedRecall, weightedF1, sum(support));
    fprintf(fid, '\\midrule\n');
    fprintf(fid, '\\textbf{Overall Accuracy} & \\multicolumn{4}{c}{%.4f} \\\\\n', accuracy);
    fprintf(fid, '\\bottomrule\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\end{table}\n');
    fclose(fid);
    fprintf('  Saved: classification_table.tex\n');

    % -----------------------------------------------------------------
    % 8. Training Configuration & Results Summary
    % -----------------------------------------------------------------
    summaryPath = fullfile(analysisDir, 'results_summary.txt');
    fid = fopen(summaryPath, 'w');
    fprintf(fid, 'Training Results Summary\n');
    fprintf(fid, '========================================\n\n');
    fprintf(fid, 'Model: %s\n', modelName);
    fprintf(fid, 'Model Type: %s\n', typeStr);
    fprintf(fid, 'Date: %s\n\n', datestr(now));

    fprintf(fid, 'Dataset:\n');
    fprintf(fid, '  Directory: %s\n', datasetDir);
    fprintf(fid, '  Train samples: %d\n', size(trainData_n, 3));
    fprintf(fid, '  Val samples: %d\n', size(valData_n, 3));
    fprintf(fid, '  Test samples: %d\n', size(testData_n, 3));
    fprintf(fid, '  Num classes: %d\n', numClasses);
    fprintf(fid, '  Frame shape: [%d, %d]\n', size(trainData_n, 1), size(trainData_n, 2));

    if isfield(datasetConfig, 'split_method')
        fprintf(fid, '  Split method: %s\n', datasetConfig.split_method);
    end
    if isfield(datasetConfig, 'joint_selection_option')
        fprintf(fid, '  Joints: %s\n', datasetConfig.joint_selection_option);
    end
    if isfield(datasetConfig, 'selection_mode')
        fprintf(fid, '  Mode: %s\n', datasetConfig.selection_mode);
    end

    fprintf(fid, '\nHyperparameters:\n');
    if modelType == 0
        fprintf(fid, '  Conv1 Filters: %d\n', Hyper_1);
        fprintf(fid, '  Conv2 Filters: %d\n', Hyper_2);
    elseif modelType == 1
        fprintf(fid, '  LSTM Hidden Units: %d\n', Hyper_1);
        fprintf(fid, '  FC Hidden Units: %d\n', Hyper_2);
    else
        fprintf(fid, '  Attention Heads: %d\n', Hyper_1);
        fprintf(fid, '  Hidden Units: %d\n', Hyper_2);
    end
    fprintf(fid, '  Dropout: %.2f\n', Hyper_3);
    fprintf(fid, '  Max Epochs: %d\n', Hyper_4);
    fprintf(fid, '  Learning Rate: %g\n', Hyper_5);
    fprintf(fid, '  Mini Batch Size: %d\n', Hyper_6);
    fprintf(fid, '  Validation Frequency: %d\n', valFreq);
    fprintf(fid, '  Validation Patience: %d\n', valPat);

    fprintf(fid, '\nResults:\n');
    fprintf(fid, '  Overall Accuracy: %.4f (%.2f%%)\n', accuracy, accuracy * 100);
    fprintf(fid, '  Macro Precision: %.4f\n', macroPrecision);
    fprintf(fid, '  Macro Recall: %.4f\n', macroRecall);
    fprintf(fid, '  Macro F1: %.4f\n', macroF1);
    fprintf(fid, '  Weighted F1: %.4f\n', weightedF1);

    fprintf(fid, '\nPer-Class Results:\n');
    for c = 1:numClassesActual
        fprintf(fid, '  %s: P=%.3f, R=%.3f, F1=%.3f, N=%d\n', ...
            classNamesList{c}, precision(c), recall(c), f1(c), support(c));
    end

    fprintf(fid, '\nFiles:\n');
    fprintf(fid, '  Model: %s\n', fullfile(mat_dir, fileName));
    fprintf(fid, '  Analysis: %s\n', analysisDir);
    fclose(fid);
    fprintf('  Saved: results_summary.txt\n');

    % -----------------------------------------------------------------
    % 9. Save confusion matrix and metrics as .mat for later use
    % -----------------------------------------------------------------
    save(fullfile(analysisDir, 'confusion_matrix.mat'), 'cm');
    save(fullfile(analysisDir, 'metrics.mat'), ...
        'precision', 'recall', 'f1', 'support', ...
        'macroPrecision', 'macroRecall', 'macroF1', ...
        'weightedPrecision', 'weightedRecall', 'weightedF1', ...
        'accuracy', 'classNamesList');

    % Save test predictions with traceability
    if hasTraceability
        save(fullfile(analysisDir, 'test_predictions.mat'), ...
            'predictedLabels', 'testLabels', 'testParticipants', 'testFilenames', 'scores');
        fprintf('  Saved: test_predictions.mat (with participant/filename traceability)\n');
    else
        save(fullfile(analysisDir, 'test_predictions.mat'), ...
            'predictedLabels', 'testLabels', 'scores');
        fprintf('  Saved: test_predictions.mat\n');
    end

    % -----------------------------------------------------------------
    % 10. Per-Participant Accuracy (if traceability available)
    % -----------------------------------------------------------------
    if hasTraceability
        uniqueParticipants = unique(testParticipants);
        numParticipants = length(uniqueParticipants);

        fig5 = figure('Visible', 'off', 'Position', [100 100 max(700, numParticipants*60) 450]);
        participantAcc = zeros(numParticipants, 1);
        participantCount = zeros(numParticipants, 1);

        for p = 1:numParticipants
            mask = strcmp(testParticipants, uniqueParticipants{p});
            participantAcc(p) = mean(predictedLabels(mask) == testLabels(mask));
            participantCount(p) = sum(mask);
        end

        bar(1:numParticipants, participantAcc, 'FaceColor', [0.3 0.6 0.8]);
        hold on;
        yline(accuracy, 'r--', 'LineWidth', 2, 'Label', sprintf('Overall: %.3f', accuracy));
        for p = 1:numParticipants
            text(p, participantAcc(p) + 0.02, ...
                sprintf('%.2f\n(%d)', participantAcc(p), participantCount(p)), ...
                'HorizontalAlignment', 'center', 'FontSize', 8);
        end

        participantLabels = cellfun(@(x) ['P' x], uniqueParticipants, 'UniformOutput', false);
        set(gca, 'XTick', 1:numParticipants, 'XTickLabel', participantLabels);
        xlabel('Participant', 'FontWeight', 'bold');
        ylabel('Accuracy', 'FontWeight', 'bold');
        title(sprintf('Per-Participant Accuracy — %s (%s)', modelName, typeStr), 'FontWeight', 'bold');
        ylim([0, 1.15]);
        grid on;
        set(gca, 'GridAlpha', 0.3);
        hold off;

        exportgraphics(fig5, fullfile(analysisDir, 'per_participant_accuracy.png'), 'Resolution', 300);
        savefig(fig5, fullfile(analysisDir, 'per_participant_accuracy.fig'));
        close(fig5);
        fprintf('  Saved: per_participant_accuracy.png\n');

        % Write per-participant breakdown to report
        partReportPath = fullfile(analysisDir, 'per_participant_report.txt');
        fid = fopen(partReportPath, 'w');
        fprintf(fid, 'Per-Participant Test Accuracy\n');
        fprintf(fid, '=============================================\n');
        fprintf(fid, '%-15s %10s %10s\n', 'Participant', 'Accuracy', 'Samples');
        fprintf(fid, '---------------------------------------------\n');
        for p = 1:numParticipants
            fprintf(fid, 'P%-14s %10.3f %10d\n', ...
                uniqueParticipants{p}, participantAcc(p), participantCount(p));
        end
        fprintf(fid, '---------------------------------------------\n');
        fprintf(fid, '%-15s %10.3f %10d\n', 'Overall', accuracy, length(testLabels));
        fprintf(fid, '=============================================\n');
        fclose(fid);
        fprintf('  Saved: per_participant_report.txt\n');
    end

    % =====================================================================
    % Print summary to console and app
    % =====================================================================
    fprintf('\n========================================\n');
    fprintf('Analysis Complete\n');
    fprintf('========================================\n');
    fprintf('  Accuracy: %.2f%%\n', accuracy * 100);
    fprintf('  Macro F1: %.4f\n', macroF1);
    fprintf('  Weighted F1: %.4f\n', weightedF1);
    fprintf('  Analysis saved to: %s\n', analysisDir);
    fprintf('========================================\n');

    app.TextArea.Value = [app.TextArea.Value; ...
        sprintf('Macro F1: %.4f', macroF1); ...
        sprintf('Analysis saved to: %s', analysisDir)];

    fprintf('Data preprocessing complete.\n');
end


% =========================================================================
% Helper: convert 3D array to cell array of sequences
% =========================================================================
function sequenceData = sequence(data)
    numSamples = size(data, 3);
    sequenceData = cell(numSamples, 1);
    for i = 1:numSamples
        sequenceData{i} = squeeze(data(:, :, i))';
    end
end