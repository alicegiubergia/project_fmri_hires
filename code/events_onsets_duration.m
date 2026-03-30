function [ev] = events_onsets_duration(filename, output_directory)
% 
% READS BEHAVIOURAL FILE TO GET TIMING OF BLOCKS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ==== Load file ====
opts = detectImportOptions(filename, 'FileType','text', 'Delimiter', '\t');
opts.DataLines = [5, Inf];
opts.VariableNamesLine = 4;
opts = setvaropts(opts, 'Var4', 'Type', 'string');
data = readtable(filename, opts);
data = data(~isnan(data.Trial), :);
data = data(~strcmp(data.Code, "10"), :);               % Remove Pulses
data = data(~strcmp(data.EventType, 'Quit'), :);        % Remove Quit
data = data(~strcmp(data.EventType, 'Response'), :);    % Remove Quit
data.Duration = str2double(data.Duration);
data = data((data.Trial ~= 0), :);                      % Remove dummies
data = data(~startsWith(data.Code, 'ITI'), :);
data = data(~startsWith(data.Code, 'blank'), :);
data.Code = regexprep(data.Code, '^[^_]+_', '');

% ==== Select variables ====
events_type = unique(data.("Code"));
time_values = (data.Time - data.Time(1)) / 10000;
duration_values = data.Duration / 10000;

% ==== Generate struct ====
names = cell(length(events_type), 1);
onsets = cell(length(events_type), 1);
durations = cell(length(events_type), 1);

for i = 1:length(events_type)
    event = events_type{i};
    mask = strcmp(data.("Code"), event);
    
    % Save values
    names{i} = event;
    onsets{i} = time_values(mask);
    durations{i} = duration_values(mask);
end

% ==== Final structure ====
ev = struct('name', names, 'onset', onsets, 'duration', durations);

% ==== Save ====
%folder = dir(filename).folder;
save(fullfile(output_directory, 'SPM_conditions.mat'), 'ev');
    