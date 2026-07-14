function processDataset(selected_joints_option, data_augmentation_methods, train_ratio, val_ratio, window_size, mode, raw_data_dir, project_dir, dataset_name)
    % Determine which joints are selected based on the user's selection
    if strcmp(selected_joints_option, 'Whole hand (25)')
        selected_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]; 
    elseif strcmp(selected_joints_option, 'Fingertips (5)')
        selected_joints = [4, 9, 14, 19, 24]; 
    elseif strcmp(selected_joints_option, 'Palm (11)')
        selected_joints = [0, 1, 2, 5, 6, 10, 11, 15, 16, 20, 21]; 
    elseif strcmp(selected_joints_option, 'Intermediate Phalanx (4)')
        selected_joints = [7, 12, 17, 22]; 
    elseif strcmp(selected_joints_option, 'Distal Phalanx (5)')
        selected_joints = [3, 8, 13, 18, 23]; 
    else
        error('Invalid joint selection.');
    end

    % Get the gesture and annotation folders
    gesture_folder = fullfile(raw_data_dir, 'gestures');
    annotation_folder = fullfile(raw_data_dir, 'annotations');
    data_output_dir = fullfile(project_dir, 'dataset', dataset_name);

    % Check if directories exist
    if ~isfolder(gesture_folder)
        error('Gesture folder does not exist: %s', gesture_folder);
    end
    if ~isfolder(annotation_folder)
        error('Annotation folder does not exist: %s', annotation_folder);
    end
    if ~exist(data_output_dir, 'dir')
        mkdir(data_output_dir);
    end
    
    % Get the list of gesture files
    gesture_files = dir(fullfile(gesture_folder, '*.csv'));
    num_files = length(gesture_files);
    
    % =====================================================================
    % Parse all filenames to extract class, id, and participant
    % Expected format: {GestureClass}-id-{N}-participant-{N}.csv
    % Splitting by '-' gives: {Class, 'id', N, 'participant', N}
    % Note: multi-word classes like "Two hands scale" use spaces, not dashes
    % =====================================================================
    file_classes = cell(num_files, 1);
    file_ids = cell(num_files, 1);
    file_participants = cell(num_files, 1);
    
    for i = 1:num_files
        [~, name, ~] = fileparts(gesture_files(i).name);
        parts = split(name, '-');
        file_classes{i} = parts{1};         % Gesture class
        file_ids{i} = parts{3};             % ID number
        file_participants{i} = parts{5};    % Participant number
    end
    
    % Get unique classes and participants
    unique_classes = unique(file_classes);
    unique_participants = unique(file_participants);
    num_classes_found = length(unique_classes);
    num_participants = length(unique_participants);
    
    fprintf('Found %d classes: ', num_classes_found);
    fprintf('%s ', unique_classes{:});
    fprintf('\n');
    fprintf('Found %d participants: ', num_participants);
    fprintf('%s ', unique_participants{:});
    fprintf('\n');
    
    % =====================================================================
    % Determine split method from data_augmentation_methods parameter
    %   'Method 1' = within-participant  (split samples per class x participant)
    %   'Method 2' = across-participant  (split participants into disjoint sets)
    % =====================================================================
    if strcmp(data_augmentation_methods, 'Within Participant')
        split_method = 'within';
        fprintf('Split method: WITHIN-participant (every participant in all splits)\n');
    elseif strcmp(data_augmentation_methods, 'Across Participant')
        split_method = 'across';
        fprintf('Split method: ACROSS-participant (disjoint participant sets)\n');
    else
        error('Invalid split method. Use ''Method 1'' (within-participant) or ''Method 2'' (across-participant).');
    end
    
    % Convert ratios from /10 scale to fractions
    train_frac = train_ratio / 10;
    val_frac = val_ratio / 10;
    test_frac = 1 - train_frac - val_frac;
    
    fprintf('Split ratio: %.0f%% train / %.0f%% val / %.0f%% test\n', ...
        train_frac*100, val_frac*100, test_frac*100);
    
    % =====================================================================
    % Assign files to train/val/test splits
    % =====================================================================
    % Each file gets assigned: 'train', 'val', or 'test'
    file_split = cell(num_files, 1);
    
    if strcmp(split_method, 'within')
        % ---- WITHIN-PARTICIPANT SPLIT ----
        % For each unique (class, participant) group, split files at ratio
        % Priority: test > val > train for small groups
        
        for c = 1:length(unique_classes)
            for p = 1:length(unique_participants)
                % Find file indices belonging to this (class, participant) group
                mask = strcmp(file_classes, unique_classes{c}) & ...
                       strcmp(file_participants, unique_participants{p});
                group_indices = find(mask);
                n = length(group_indices);
                
                if n == 0
                    continue;
                end
                
                % Shuffle
                group_indices = group_indices(randperm(n));
                
                % Compute split counts with test-first priority
                [n_test, n_val, n_train] = compute_split_counts(n, test_frac, val_frac);
                
                % Assign splits
                for k = 1:n_test
                    file_split{group_indices(k)} = 'test';
                end
                for k = (n_test+1):(n_test+n_val)
                    file_split{group_indices(k)} = 'val';
                end
                for k = (n_test+n_val+1):n
                    file_split{group_indices(k)} = 'train';
                end
            end
        end
        
    else
        % ---- ACROSS-PARTICIPANT SPLIT ----
        % Split participants into disjoint train/val/test groups
        % All data from a participant goes to exactly one split
        
        shuffled_participants = unique_participants(randperm(num_participants));
        
        % Compute participant split counts with test-first priority
        [np_test, np_val, ~] = compute_split_counts(num_participants, test_frac, val_frac);
        
        test_participants = shuffled_participants(1:np_test);
        val_participants = shuffled_participants(np_test+1:np_test+np_val);
        train_participants = shuffled_participants(np_test+np_val+1:end);
        
        fprintf('  Train participants (%d): ', length(train_participants));
        fprintf('%s ', train_participants{:});
        fprintf('\n');
        fprintf('  Val   participants (%d): ', length(val_participants));
        fprintf('%s ', val_participants{:});
        fprintf('\n');
        fprintf('  Test  participants (%d): ', length(test_participants));
        fprintf('%s ', test_participants{:});
        fprintf('\n');
        
        % Assign each file based on its participant
        for i = 1:num_files
            if ismember(file_participants{i}, test_participants)
                file_split{i} = 'test';
            elseif ismember(file_participants{i}, val_participants)
                file_split{i} = 'val';
            else
                file_split{i} = 'train';
            end
        end
    end
    
    % Collect indices per split
    train_indices = find(strcmp(file_split, 'train'));
    val_indices = find(strcmp(file_split, 'val'));
    test_indices = find(strcmp(file_split, 'test'));
    
    total_train = length(train_indices);
    total_val = length(val_indices);
    total_test = length(test_indices);
    
    fprintf('\nSplit result: %d train, %d val, %d test (%d total)\n', ...
        total_train, total_val, total_test, total_train + total_val + total_test);
    
    % Print per-class breakdown
    fprintf('\nPer-class breakdown:\n');
    fprintf('  %-25s %8s %8s %8s %8s\n', 'Class', 'Train', 'Val', 'Test', 'Total');
    fprintf('  %s\n', repmat('-', 1, 57));
    for c = 1:length(unique_classes)
        cls = unique_classes{c};
        n_tr = sum(strcmp(file_classes(train_indices), cls));
        n_va = sum(strcmp(file_classes(val_indices), cls));
        n_te = sum(strcmp(file_classes(test_indices), cls));
        fprintf('  %-25s %8d %8d %8d %8d\n', cls, n_tr, n_va, n_te, n_tr+n_va+n_te);
    end
    
    % =====================================================================
    % Preallocate datasets
    % =====================================================================
    frame_size = calculate_frame_size(selected_joints, mode);
    
    trainData = zeros(window_size, frame_size, total_train);
    valData = zeros(window_size, frame_size, total_val);
    testData = zeros(window_size, frame_size, total_test);
    
    trainLabels = cell(total_train, 1);
    valLabels = cell(total_val, 1);
    testLabels = cell(total_test, 1);
    
    trainParticipants = cell(total_train, 1);
    valParticipants = cell(total_val, 1);
    testParticipants = cell(total_test, 1);
    
    trainFilenames = cell(total_train, 1);
    valFilenames = cell(total_val, 1);
    testFilenames = cell(total_test, 1);
    
    % =====================================================================
    % Process files for each split
    % =====================================================================
    [trainData, trainLabels, trainParticipants, trainFilenames] = ...
        process_files(gesture_folder, annotation_folder, train_indices, ...
        gesture_files, file_classes, file_ids, file_participants, ...
        window_size, selected_joints, mode, trainData, trainLabels, ...
        trainParticipants, trainFilenames, 'train');
    
    [valData, valLabels, valParticipants, valFilenames] = ...
        process_files(gesture_folder, annotation_folder, val_indices, ...
        gesture_files, file_classes, file_ids, file_participants, ...
        window_size, selected_joints, mode, valData, valLabels, ...
        valParticipants, valFilenames, 'val');
    
    [testData, testLabels, testParticipants, testFilenames] = ...
        process_files(gesture_folder, annotation_folder, test_indices, ...
        gesture_files, file_classes, file_ids, file_participants, ...
        window_size, selected_joints, mode, testData, testLabels, ...
        testParticipants, testFilenames, 'test');

    % Save original labels (before categorical conversion)
    save(fullfile(data_output_dir, 'trainLabels_og.mat'), 'trainLabels');
    save(fullfile(data_output_dir, 'valLabels_og.mat'), 'valLabels');
    save(fullfile(data_output_dir, 'testLabels_og.mat'), 'testLabels');

    % Save participant and filename traceability info
    save(fullfile(data_output_dir, 'trainParticipants.mat'), 'trainParticipants');
    save(fullfile(data_output_dir, 'valParticipants.mat'), 'valParticipants');
    save(fullfile(data_output_dir, 'testParticipants.mat'), 'testParticipants');
    save(fullfile(data_output_dir, 'trainFilenames.mat'), 'trainFilenames');
    save(fullfile(data_output_dir, 'valFilenames.mat'), 'valFilenames');
    save(fullfile(data_output_dir, 'testFilenames.mat'), 'testFilenames');

    % Calculate mean and standard deviation of the training data
    meanSkeleton = mean(trainData(:));
    stdSkeleton = std(trainData(:));

    fprintf('Mean is %d\n', meanSkeleton);
    fprintf('Std is %d\n', stdSkeleton);
    
    % Normalize the data using training statistics
    trainData_n = (trainData - meanSkeleton) / stdSkeleton;
    valData_n = (valData - meanSkeleton) / stdSkeleton;
    testData_n = (testData - meanSkeleton) / stdSkeleton; 

    % Convert labels to categorical
    trainLabels = categorical(trainLabels);
    valLabels = categorical(valLabels);
    testLabels = categorical(testLabels);
    
    % Save the normalised data and categorical labels
    save(fullfile(data_output_dir, 'trainData.mat'), 'trainData_n');
    save(fullfile(data_output_dir, 'trainLabels.mat'), 'trainLabels');
    save(fullfile(data_output_dir, 'valData.mat'), 'valData_n');
    save(fullfile(data_output_dir, 'valLabels.mat'), 'valLabels');
    save(fullfile(data_output_dir, 'testData.mat'), 'testData_n');
    save(fullfile(data_output_dir, 'testLabels.mat'), 'testLabels');

    numClasses = size(unique(trainLabels), 1);

    save_customisation_to_json(selected_joints_option, mode, data_augmentation_methods, ...
        window_size, train_ratio, val_ratio, numClasses, project_dir, dataset_name, ...
        meanSkeleton, stdSkeleton);

    fprintf('Data preprocessing complete.\n');
end


% =========================================================================
% Split count computation with test-first priority
% =========================================================================
function [n_test, n_val, n_train] = compute_split_counts(n, test_frac, val_frac)
    if n == 0
        n_test = 0; n_val = 0; n_train = 0;
        return;
    end
    if n == 1
        n_test = 1; n_val = 0; n_train = 0;
        return;
    end
    if n == 2
        n_test = 1; n_val = 1; n_train = 0;
        return;
    end
    
    n_test = max(1, round(n * test_frac));
    n_val = max(1, round(n * val_frac));
    
    while n_test + n_val >= n && n_val > 1
        n_val = n_val - 1;
    end
    while n_test + n_val >= n && n_test > 1
        n_test = n_test - 1;
    end
    
    n_train = n - n_test - n_val;
end


% =========================================================================
% Process files and add them to the corresponding dataset
% =========================================================================
function [dataset, labels, participants, filenames] = process_files( ...
    gesture_folder, annotation_folder, file_indices, gesture_files, ...
    file_classes, file_ids, file_participants, ...
    window_size, selected_joints, mode, ...
    dataset, labels, participants, filenames, type)
    
    parent_joints = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23];
    fingertip_joints = [4, 9, 14, 19, 24];
    index_offset = 7;
    lh_offset = 1;
    rh_offset = 25 * 7 + 1;

    count = 0;
    
    for i = 1:length(file_indices)
        idx = file_indices(i);
        gesture_file = gesture_files(idx).name;
        gesture_class = file_classes{idx};
        id_number = file_ids{idx};
        participant = file_participants{idx};
        
        gesture_data = readmatrix(fullfile(gesture_folder, gesture_file));
        
        annotation_file = fullfile(annotation_folder, [gesture_class, '.csv']);
        if ~isfile(annotation_file)
            fprintf('Warning: annotation file not found for class %s, skipping\n', gesture_class);
            continue;
        end
        annotations = readmatrix(annotation_file);
        
        row = annotations(annotations(:,1) == str2double(id_number), :);
        
        if isempty(row)
            fprintf('Warning: no annotation found for id %s in class %s, using full sequence\n', ...
                id_number, gesture_class);
            start_frame = 1;
            end_frame = size(gesture_data, 1);
            if end_frame - start_frame + 1 < window_size
                frames = [gesture_data(start_frame:end_frame, :); ...
                          repmat(gesture_data(end_frame, :), ...
                          window_size - (end_frame - start_frame + 1), 1)];
            else
                mid = floor((start_frame + end_frame) / 2);
                s = max(1, mid - floor(window_size/2));
                frames = gesture_data(s:s+window_size-1, :);
            end
        elseif size(row, 2) == 3
            middle_frame = row(3);
            start_frame = max(1, middle_frame - floor(window_size / 2));
            end_frame = min(size(gesture_data, 1), start_frame + window_size - 1);
            
            if end_frame - start_frame + 1 < window_size
                if start_frame == 1
                    frames = [repmat(gesture_data(start_frame, :), ...
                              window_size - (end_frame - start_frame + 1), 1); ...
                              gesture_data(start_frame:end_frame, :)];
                else
                    frames = [gesture_data(start_frame:end_frame, :); ...
                              repmat(gesture_data(end_frame, :), ...
                              window_size - (end_frame - start_frame + 1), 1)];
                end
            else
                frames = gesture_data(start_frame:end_frame, :);
            end

        elseif size(row, 2) > 3
            start_frame = max(1, row(3));
            end_frame = row(4);
            if end_frame - start_frame + 1 < window_size
                frames = [repmat(gesture_data(start_frame, :), ...
                          window_size - (end_frame - start_frame + 1), 1); ...
                          gesture_data(start_frame:end_frame, :)];
            else
                frames = gesture_data(start_frame:min(start_frame + window_size - 1, ...
                         size(gesture_data, 1)), :);
            end
        end
        
        frames_local = convert_to_local(frames, window_size, lh_offset, rh_offset, ...
                                        index_offset, parent_joints, fingertip_joints);
        processed_frames = process_frames_based_on_mode(frames_local, selected_joints, mode);

        fprintf('Processed %d/%d files of the %s class for %s\n', ...
            i, length(file_indices), gesture_class, type);
        
        count = count + 1;
        dataset(:,:,count) = processed_frames;
        labels{count} = gesture_class;
        participants{count} = participant;
        filenames{count} = gesture_file;
    end
end


% =========================================================================
% Local coordinate conversion
% =========================================================================
function data_buffer_local = convert_to_local(data_buffer, n_frames, ...
    lh_offset, rh_offset, index_offset, parent_joints, fingertip_joints)
    
    data_buffer_local = data_buffer;
    lh_wrist_pos = data_buffer(:, lh_offset + (0:2));
    rh_wrist_pos = data_buffer(:, rh_offset + (0:2));

    for frame = 1:n_frames
        for joint = 0:24
            parent_joint = parent_joints(joint + 1);
            
            lh_global_pos = data_buffer(frame, lh_offset + index_offset * joint + (0:2));
            rh_global_pos = data_buffer(frame, rh_offset + index_offset * joint + (0:2));
            lh_local_pos = lh_global_pos - lh_wrist_pos(frame, :);
            rh_local_pos = rh_global_pos - rh_wrist_pos(frame, :);
            data_buffer_local(frame, lh_offset + index_offset * joint + (0:2)) = lh_local_pos;
            data_buffer_local(frame, rh_offset + index_offset * joint + (0:2)) = rh_local_pos;

            if parent_joint == -1
                continue;
            elseif ismember(joint, fingertip_joints)
                lh_quat_start = lh_offset + index_offset * joint;
                rh_quat_start = rh_offset + index_offset * joint;
                data_buffer_local(frame, lh_quat_start + (3:5)) = [0, 0, 0];
                data_buffer_local(frame, lh_quat_start + 6) = 1;
                data_buffer_local(frame, rh_quat_start + (3:5)) = [0, 0, 0];
                data_buffer_local(frame, rh_quat_start + 6) = 1;
            else
                lh_parent_quat = data_buffer(frame, lh_offset + index_offset * parent_joint + [6, 3:5]);
                rh_parent_quat = data_buffer(frame, rh_offset + index_offset * parent_joint + [6, 3:5]);
                lh_global_quat = data_buffer(frame, lh_offset + index_offset * joint + [6, 3:5]);
                rh_global_quat = data_buffer(frame, rh_offset + index_offset * joint + [6, 3:5]);
                lh_parent_quat_inv = quatconj(lh_parent_quat);
                rh_parent_quat_inv = quatconj(rh_parent_quat);
                lh_local_quat = quatmultiply(lh_parent_quat_inv, lh_global_quat);
                rh_local_quat = quatmultiply(rh_parent_quat_inv, rh_global_quat);
                data_buffer_local(frame, lh_offset + index_offset * joint + [6, 3:5]) = lh_local_quat;
                data_buffer_local(frame, rh_offset + index_offset * joint + [6, 3:5]) = rh_local_quat;
            end
        end
    end
    
    for frame = 1:n_frames
        data_buffer_local(frame, lh_offset + (0:2)) = [0, 0, 0];
        data_buffer_local(frame, rh_offset + (0:2)) = [0, 0, 0];
    end
end


% =========================================================================
% Calculate frame size based on selected mode and joints
% =========================================================================
function frame_size = calculate_frame_size(selected_joints, mode)
    num_joints = length(selected_joints);
    if mode == 0
        frame_size = 2 * 3 * num_joints;
    elseif mode == 1
        frame_size = 2 * 4 * num_joints;
    else
        frame_size = 2 * 7 * num_joints;
    end
end


% =========================================================================
% Process frames based on selected mode
% =========================================================================
function processed_frames = process_frames_based_on_mode(frames_local, selected_joints, mode)
    num_joints = length(selected_joints);
    
    if mode == 0
        num_columns = 2 * 3 * num_joints;
    elseif mode == 1
        num_columns = 2 * 4 * num_joints;
    else
        num_columns = 2 * 7 * num_joints;
    end
    
    num_frames = size(frames_local, 1);
    processed_frames = zeros(num_frames, num_columns);
    col_idx = 1;
    
    for joint = selected_joints
        lh_position_indices = joint * 7 + (1:3);
        lh_rotation_indices = joint * 7 + (4:7);
        if mode == 0
            processed_frames(:, col_idx:col_idx+2) = frames_local(:, lh_position_indices);
            col_idx = col_idx + 3;
        elseif mode == 1
            processed_frames(:, col_idx:col_idx+3) = frames_local(:, lh_rotation_indices);
            col_idx = col_idx + 4;
        else
            processed_frames(:, col_idx:col_idx+2) = frames_local(:, lh_position_indices);
            processed_frames(:, col_idx+3:col_idx+6) = frames_local(:, lh_rotation_indices);
            col_idx = col_idx + 7;
        end
    end
    
    for joint = selected_joints
        rh_position_indices = 25 * 7 + joint * 7 + (1:3);
        rh_rotation_indices = 25 * 7 + joint * 7 + (4:7);
        if mode == 0
            processed_frames(:, col_idx:col_idx+2) = frames_local(:, rh_position_indices);
            col_idx = col_idx + 3;
        elseif mode == 1
            processed_frames(:, col_idx:col_idx+3) = frames_local(:, rh_rotation_indices);
            col_idx = col_idx + 4;
        else
            processed_frames(:, col_idx:col_idx+2) = frames_local(:, rh_position_indices);
            processed_frames(:, col_idx+3:col_idx+6) = frames_local(:, rh_rotation_indices);
            col_idx = col_idx + 7;
        end
    end
end


% =========================================================================
% Save customisation details to JSON
% =========================================================================
function save_customisation_to_json(selected_joints_option, mode, data_augmentation_methods, ...
    window_size, train_ratio, val_ratio, numClasses, project_dir, dataset_name, ...
    meanSkeleton, stdSkeleton)
    
    customisation_details = struct();
    customisation_details.dataset_name = dataset_name;
    customisation_details.dataset_directory = fullfile(project_dir, 'dataset', dataset_name);
    customisation_details.joint_selection_option = selected_joints_option;
    
    if mode == 0
        customisation_details.selection_mode = 'Position only';
    elseif mode == 1
        customisation_details.selection_mode = 'Rotation only';
    else
        customisation_details.selection_mode = 'Both';
    end
    
    if strcmp(data_augmentation_methods, 'Within Participant')
        customisation_details.split_method = 'within-participant';
    elseif strcmp(data_augmentation_methods, 'Across Participant')
        customisation_details.split_method = 'across-participant';
    else
        customisation_details.split_method = data_augmentation_methods;
    end
    
    customisation_details.window_size = window_size;
    customisation_details.train_ratio = train_ratio;
    customisation_details.val_ratio = val_ratio;
    customisation_details.test_ratio = 10 - train_ratio - val_ratio;
    customisation_details.numClasses = numClasses;
    customisation_details.meanSkeleton = meanSkeleton;
    customisation_details.stdSkeleton = stdSkeleton;
    
    json_str = jsonencode(customisation_details);
    
    jsonFileName = 'dataset_customisation.json';
    json_file_path = fullfile(customisation_details.dataset_directory, jsonFileName);
    
    fid = fopen(json_file_path, 'w');
    if fid == -1
        error('Cannot create JSON file: %s', json_file_path);
    end
    fwrite(fid, json_str, 'char');
    fclose(fid);
    
    fprintf('Customisation details saved to: %s\n', json_file_path);
    
    project_description_dir = fullfile(project_dir, 'project.json');
    
    if isfile(project_description_dir)
        fid = fopen(project_description_dir, 'r');
        rawJsonText = fread(fid, '*char')';
        fclose(fid);
        projectInfo = jsondecode(rawJsonText);
    else
        projectInfo = struct();
    end
    
    if ~isfield(projectInfo, 'DataPreprocessing') || ~isstruct(projectInfo.DataPreprocessing)
        projectInfo.DataPreprocessing = struct();
    end
    
    if ~isfield(projectInfo.DataPreprocessing, 'dataset_name') || ~iscell(projectInfo.DataPreprocessing.dataset_name)
        projectInfo.DataPreprocessing.dataset_name = {};
    end
    
    newDatasetName = dataset_name;
    if ~ismember(newDatasetName, projectInfo.DataPreprocessing.dataset_name)
        projectInfo.DataPreprocessing.dataset_name{end+1} = newDatasetName;
    end
    
    projectInfo.LastModifiedOn = datetime('now');
    
    jsonText = jsonencode(projectInfo);
    fid = fopen(project_description_dir, 'w');
    fwrite(fid, jsonText, 'char');
    fclose(fid);

    fprintf('Modified project JSON file\n');
end