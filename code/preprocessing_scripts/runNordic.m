%% Run NORDIC on already existing files
%data_path='/home/local/IMAGO7/agiubergia/medea/test_20250407/converted/func/';
data_path = '~/medea/progetto_reading/derivatives/mypipeline/sub-00/func/';
sessions = dir(data_path);

for i = 3:length(sessions)
    fprintf('Running session %s\n', sessions(i).name);
    session_folder = fullfile(data_path, sessions(i).name);
    output_folder = fullfile(session_folder, 'nordic/');
    mag_file = fullfile(session_folder, 'epi.nii.gz');
    NIFTI_NORDIC_batch('', '', mag_file, fullfile(output_folder, 'epi_nordic.nii.gz'))
end
    

