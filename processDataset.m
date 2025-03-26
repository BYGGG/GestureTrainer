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
    
    % Precalculate the total number of labels based on ratio split
    total_train_labels = 0;
    total_val_labels = 0;
    total_test_labels = 0;

    % Group files by gesture class
    gesture_class_map = containers.Map;
    for i = 1:num_files
        gesture_file = gesture_files(i).name;
        parts = split(gesture_file, '-');
        gesture_class = parts{2};
        if isKey(gesture_class_map, gesture_class)
            gesture_class_map(gesture_class) = [gesture_class_map(gesture_class), i];
        else
            gesture_class_map(gesture_class) = i;
        end
    end

    gesture_classes = keys(gesture_class_map);
    for c = 1:length(gesture_classes)
        file_indices = gesture_class_map(gesture_classes{c});
        num_class_files = length(file_indices);

        % Calculate the number of files for train, validation, and test
        num_train = round((train_ratio / 10) * num_class_files);
        num_val = round((val_ratio / 10) * num_class_files);
        num_test = num_class_files - num_train - num_val;

        % Update the total labels count
        total_train_labels = total_train_labels + num_train;
        total_val_labels = total_val_labels + num_val;
        total_test_labels = total_test_labels + num_test;
    end
    
    % Preallocate the size for labels
    trainLabels = cell(total_train_labels, 1);
    valLabels = cell(total_val_labels, 1);
    testLabels = cell(total_test_labels, 1);
    
    % Estimate frame data size for preallocation based on the first file
    if num_files > 0
        frame_size = calculate_frame_size(selected_joints, mode);
        
        % Preallocate space for train, val, and test datasets
        trainData = zeros(window_size, frame_size, total_train_labels);
        valData = zeros(window_size, frame_size, total_val_labels);
        testData = zeros(window_size, frame_size, total_test_labels);
    end
    
    % Initialize label counts for adding processed data
    trainLabelCount = 0;
    valLabelCount = 0;
    testLabelCount = 0;
    
    % Process each gesture class and split the data
    for c = 1:length(gesture_classes)
        class_name = gesture_classes{c};
        file_indices = gesture_class_map(class_name);
        num_class_files = length(file_indices);
        
        % Shuffle the file indices
        file_indices = file_indices(randperm(num_class_files));
        
        % Calculate split counts
        num_train = round((train_ratio / 10) * num_class_files);
        num_val = round((val_ratio / 10) * num_class_files);

        % Split the indices
        train_files = file_indices(1:num_train);
        val_files = file_indices(num_train + 1:num_train + num_val);
        test_files = file_indices(num_train + num_val + 1:end);

        % Process files for train, validation, and test
        [trainData, trainLabelCount, trainLabels] = process_files(gesture_folder, annotation_folder, train_files, gesture_files, class_name, window_size, selected_joints, mode, trainData, trainLabelCount, trainLabels, 'train');
        [valData, valLabelCount, valLabels] = process_files(gesture_folder, annotation_folder, val_files, gesture_files, class_name, window_size, selected_joints, mode, valData, valLabelCount, valLabels, 'val');
        [testData, testLabelCount, testLabels] = process_files(gesture_folder, annotation_folder, test_files, gesture_files, class_name, window_size, selected_joints, mode, testData, testLabelCount, testLabels, 'test');
    end

    % Save the data and labels
    save(fullfile(data_output_dir, 'trainLabels_og.mat'), 'trainLabels');
    save(fullfile(data_output_dir, 'valLabels_og.mat'), 'valLabels');
    save(fullfile(data_output_dir, 'testLabels_og.mat'), 'testLabels');

    % Calculate mean and standard deviation of the training data
    meanSkeleton = mean(trainData(:));
    stdSkeleton = std(trainData(:));

    fprintf('Mean is %d\n', meanSkeleton);
    fprintf('Std is %d\n', stdSkeleton);
    
    % Normalize the training data
    trainData_n = (trainData - meanSkeleton) / stdSkeleton;
    valData_n = (valData - meanSkeleton) / stdSkeleton;
    testData_n = (testData - meanSkeleton) / stdSkeleton; 

    % Convert labels to categorical
    trainLabels = categorical(trainLabels);
    valLabels = categorical(valLabels);
    testLabels = categorical(testLabels);
    
    % Save the data and labels
    save(fullfile(data_output_dir, 'trainData.mat'), 'trainData_n');
    save(fullfile(data_output_dir, 'trainLabels.mat'), 'trainLabels');
    save(fullfile(data_output_dir, 'valData.mat'), 'valData_n');
    save(fullfile(data_output_dir, 'valLabels.mat'), 'valLabels');
    save(fullfile(data_output_dir, 'testData.mat'), 'testData_n');
    save(fullfile(data_output_dir, 'testLabels.mat'), 'testLabels');

    numClasses = size(unique(trainLabels), 1);

    save_customisation_to_json(selected_joints_option, mode, data_augmentation_methods, window_size, train_ratio, val_ratio, numClasses, project_dir, dataset_name);

    fprintf('Data preprocessing complete.\n');
end

% Function to process files and add them to the corresponding dataset
function [dataset, labelCount, labels] = process_files(gesture_folder, annotation_folder, file_indices, gesture_files, gesture_class, window_size, selected_joints, mode, dataset, labelCount, labels, type)
    % Define parent-child relationships for joints
    parent_joints = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23];
    index_offset = 7;  % Joint data offsets (for 3 position, 4 orientation)
    lh_offset = 1;     % Left hand starts at offset 1
    rh_offset = 25 * 7 + 1;  % Right hand starts after left hand's joints

    for i = 1:length(file_indices)
        index = file_indices(i);
        gesture_file = gesture_files(index).name;
        gesture_data = readmatrix(fullfile(gesture_folder, gesture_file));
        
        % Extract the id_number from the filename
        [~, name, ~] = fileparts(gesture_file);
        parts = split(name, '-');
        id_number = parts{4};
        
        % Load the corresponding annotation file
        annotation_file = fullfile(annotation_folder, [gesture_class, '.csv']);
        if ~isfile(annotation_file)
            continue;
        end
        annotations = readmatrix(annotation_file);
        
        % Find the matching annotation rows by id_number
        row = annotations(annotations(:,1) == str2double(id_number), :);

        if size(row, 2) == 3
            % First kind of annotation
            middle_frame = row(3);
            start_frame = max(1, middle_frame - floor(window_size / 2));
            end_frame = min(size(gesture_data, 1), start_frame + window_size - 1);
            
            if end_frame - start_frame + 1 < window_size
                if start_frame == 1
                    frames = [repmat(gesture_data(start_frame, :), window_size - (end_frame - start_frame + 1), 1); gesture_data(start_frame:end_frame, :)];
                else
                    frames = [gesture_data(start_frame:end_frame, :); repmat(gesture_data(end_frame, :), window_size - (end_frame - start_frame + 1), 1)];
                end
            else
                frames = gesture_data(start_frame:end_frame, :);
            end

        elseif size(row, 2) > 3
            % Second kind of annotation
            start_frame = max(1, row(3));
            end_frame = row(4);
            if end_frame - start_frame + 1 < window_size
                frames = [repmat(gesture_data(start_frame, :), window_size - (end_frame - start_frame + 1), 1); gesture_data(start_frame:end_frame, :)];
            else
                frames = gesture_data(start_frame:min(start_frame + window_size - 1, size(gesture_data, 1)), :);
            end
        end
        
        % Convert to local coordinates
        frames_local = convert_to_local(frames, window_size, lh_offset, rh_offset, index_offset, parent_joints);

        % Process the frames based on the selected mode (Position, Rotation, or Both)
        processed_frames = process_frames_based_on_mode(frames_local, selected_joints, mode);

        fprintf('Processed %d files of the %s class for %s\n', i, gesture_class, type);
        
        % Add the processed frames to the dataset
        labelCount = labelCount + 1;
        dataset(:,:,labelCount) = processed_frames;
        labels{labelCount} = gesture_class;
    end
end

% Function to calculate frame size based on selected mode and joints
function frame_size = calculate_frame_size(selected_joints, mode)
    num_joints = length(selected_joints);
    if mode == 0
        frame_size = 2 * 3 * num_joints;  % Position only for both hands
    elseif mode == 1
        frame_size = 2 * 4 * num_joints;  % Rotation only for both hands
    else
        frame_size = 2 * 7 * num_joints;  % Position + Rotation for both hands
    end
end

% Function to process frames based on selected mode
function processed_frames = process_frames_based_on_mode(frames_local, selected_joints, mode)
    % Pre-determine the number of columns to allocate based on mode and selected joints
    num_joints = length(selected_joints);
    
    % Calculate the number of columns for the output
    if mode == 0
        % Position only
        num_columns = 2 * 3 * num_joints;  % 3 columns per joint per hand (left + right)
    elseif mode == 1
        % Rotation only
        num_columns = 2 * 4 * num_joints;  % 4 columns per joint per hand (left + right)
    else
        % Position + Rotation
        num_columns = 2 * 7 * num_joints;  % 7 columns per joint per hand (left + right)
    end
    
    % Preallocate processed_frames based on the number of frames (rows) and calculated columns
    num_frames = size(frames_local, 1);
    processed_frames = zeros(num_frames, num_columns);
    
    col_idx = 1;
    
    % First process left hand
    for joint = selected_joints
        % Left hand
        lh_position_indices = joint * 7 + (1:3);  % Extract position indices for left hand
        lh_rotation_indices = joint * 7 + (4:7);  % Extract rotation indices for left hand
        
        if mode == 0
            % Extract position data only for left hand
            processed_frames(:, col_idx:col_idx+2) = frames_local(:, lh_position_indices);
            col_idx = col_idx + 3;  % Move to the next column set
        elseif mode == 1
            % Extract rotation data only for left hand
            processed_frames(:, col_idx:col_idx+3) = frames_local(:, lh_rotation_indices);
            col_idx = col_idx + 4;  % Move to the next column set
        else
            % Extract both position and rotation data for left hand
            processed_frames(:, col_idx:col_idx+2) = frames_local(:, lh_position_indices);
            processed_frames(:, col_idx+3:col_idx+6) = frames_local(:, lh_rotation_indices);
            col_idx = col_idx + 7;  % Move to the next column set
        end
    end
    
    % Then process right hand
    for joint = selected_joints
        % Right hand
        rh_position_indices = 25 * 7 + joint * 7 + (1:3);  % Extract position indices for right hand
        rh_rotation_indices = 25 * 7 + joint * 7 + (4:7);  % Extract rotation indices for right hand
        
        if mode == 0
            % Extract position data only for right hand
            processed_frames(:, col_idx:col_idx+2) = frames_local(:, rh_position_indices);
            col_idx = col_idx + 3;  % Move to the next column set
        elseif mode == 1
            % Extract rotation data only for right hand
            processed_frames(:, col_idx:col_idx+3) = frames_local(:, rh_rotation_indices);
            col_idx = col_idx + 4;  % Move to the next column set
        else
            % Extract both position and rotation data for right hand
            processed_frames(:, col_idx:col_idx+2) = frames_local(:, rh_position_indices);
            processed_frames(:, col_idx+3:col_idx+6) = frames_local(:, rh_rotation_indices);
            col_idx = col_idx + 7;  % Move to the next column set
        end
    end
end

% Function to convert to local coordinates
function data_buffer_local = convert_to_local(data_buffer, n_frames, lh_offset, rh_offset, index_offset, parent_joints)
    data_buffer_local = data_buffer;

    % Extract wrist position
    lh_wrist_pos = data_buffer(:, lh_offset + (0:2));
    rh_wrist_pos = data_buffer(:, rh_offset + (0:2));

    for frame = 1:n_frames
        for joint = 0:24
            parent_joint = parent_joints(joint + 1);
            
            % Convert position to local coordinates (relative to wrist)
            lh_global_pos = data_buffer(frame, lh_offset + index_offset * joint + (0:2));
            rh_global_pos = data_buffer(frame, rh_offset + index_offset * joint + (0:2));
            lh_local_pos = lh_global_pos - lh_wrist_pos(frame, :);
            rh_local_pos = rh_global_pos - rh_wrist_pos(frame, :);
            data_buffer_local(frame, lh_offset + index_offset * joint + (0:2)) = lh_local_pos;
            data_buffer_local(frame, rh_offset + index_offset * joint + (0:2)) = rh_local_pos;

            if parent_joint == -1
                % Root joint (wrist), no quaternion transformation needed
                continue;
            end

            % Extract parent quaternion with correct order
            lh_parent_quat = data_buffer(frame, lh_offset + index_offset * parent_joint + [6, 3:5]);
            rh_parent_quat = data_buffer(frame, rh_offset + index_offset * parent_joint + [6, 3:5]);
            
            % Convert quaternion to local coordinates (relative to parent joint)
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

function save_customisation_to_json(selected_joints_option, mode, data_augmentation_methods, window_size, train_ratio, val_ratio, numClasses, project_dir, dataset_name)
    % Create a struct to hold all the details
    customisation_details = struct();

    customisation_details.dataset_name = dataset_name;
    customisation_details.dataset_directory = fullfile(project_dir, 'dataset', dataset_name);
    
    % Store the joint selection option and the actual selected joints
    customisation_details.joint_selection_option = selected_joints_option;
    
    % Store position and rotation selection
    if mode == 0
        customisation_details.selection_mode = 'Position only';
    elseif mode == 1
        customisation_details.selection_mode = 'Rotation only';
    else
        customisation_details.selection_mode = 'Both';
    end
    
    % Store the data augmentation methods
    customisation_details.data_augmentation_methods = data_augmentation_methods;
    
    % Store the window size
    customisation_details.window_size = window_size;
    
    % Store the train, validation, and test ratios
    customisation_details.train_ratio = train_ratio;
    customisation_details.val_ratio = val_ratio;
    customisation_details.test_ratio = 10 - train_ratio - val_ratio;

    customisation_details.numClasses = numClasses;
    
    % Convert the struct to JSON format
    json_str = jsonencode(customisation_details);
    
    % Define the output JSON file path
    jsonFileName = 'dataset_customisation.json';
    json_file_path = fullfile(customisation_details.dataset_directory, jsonFileName);
    
    % Write the JSON string to the file
    fid = fopen(json_file_path, 'w');
    if fid == -1
        error('Cannot create JSON file: %s', json_file_path);
    end
    fwrite(fid, json_str, 'char');
    fclose(fid);
    
    fprintf('customisation details saved to: %s\n', json_file_path);
    
    % Define the path to the project.json file
    project_description_dir = fullfile(project_dir, 'project.json');
    
    % Load the existing project information from the JSON file
    if isfile(project_description_dir)
        % Read the existing JSON file content
        fid = fopen(project_description_dir, 'r');
        rawJsonText = fread(fid, '*char')';
        fclose(fid);
        
        % Decode the JSON text into a MATLAB structure
        projectInfo = jsondecode(rawJsonText);
    else
        % If the file does not exist, initialize an empty structure
        projectInfo = struct();
    end
    
    % Ensure the DataPreprocessing field exists and is a structure
    if ~isfield(projectInfo, 'DataPreprocessing') || ~isstruct(projectInfo.DataPreprocessing)
        projectInfo.DataPreprocessing = struct();
    end
    
    % Ensure the dataset_name field exists and is a cell array
    if ~isfield(projectInfo.DataPreprocessing, 'dataset_name') || ~iscell(projectInfo.DataPreprocessing.dataset_name)
        projectInfo.DataPreprocessing.dataset_name = {};
    end
    
    % Define the new dataset name
    newDatasetName = dataset_name;
    
    % Check if the new dataset name already exists in the dataset_name list
    if ~ismember(newDatasetName, projectInfo.DataPreprocessing.dataset_name)
        % If it doesn't exist, add it to the list
        projectInfo.DataPreprocessing.dataset_name{end+1} = newDatasetName;
    end
    
    % Update the LastModifiedOn field with the current date and time
    projectInfo.LastModifiedOn = datetime('now');
    
    % Encode the modified project information as JSON
    jsonText = jsonencode(projectInfo);
    
    % Write the modified JSON back to the file
    fid = fopen(project_description_dir, 'w');
    fwrite(fid, jsonText, 'char');
    fclose(fid);

    fprintf('Modified project JSON file\n');
end
