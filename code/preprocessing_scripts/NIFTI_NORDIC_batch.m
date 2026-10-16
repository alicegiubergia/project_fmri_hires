function NIFTI_NORDIC_batch(real_file, imag_file, mag_file, output_file)
    % NIFTI_NORDIC_batch applies NORDIC denoising to NIfTI-format fMRI data.
    %
    % DESCRIPTION:
    % This function takes in real, imaginary, and magnitude NIfTI files, 
    % computes a phase image (if real and imaginary inputs are provided), 
    % and applies the NORDIC denoising algorithm to the magnitude image.
    % The output is saved to the specified output folder.
    %
    % Phase image generation uses FSL tools (fslcomplex, fslcpgeom).
    % The denoising step uses the NIFTI_NORDIC implementation from the 
    % NORDIC_Raw repository.
    %
    % USAGE:
    %   NIFTI_NORDIC_batch(real_file, imag_file, mag_file, output_folder)
    %
    % INPUTS:
    %   real_file      - Full path to the real part NIfTI file (.nii.gz)
    %   imag_file      - Full path to the imaginary part NIfTI file (.nii.gz)
    %   mag_file       - Full path to the magnitude NIfTI file (.nii.gz)
    %   output_file    - Full path to the denoised NIfTI file (.nii.gz)
    %
    % OUTPUT:
    %   Saves denoised magnitude image (and optional phase image) in the output folder.
    %
    % REQUIREMENTS:
    %   - FSL must be installed and accessible from the system path.
    %   - NORDIC_Raw and NIfTI toolboxes must be available and their paths added.
    %
    % AUTHOR:
    %   (Alice Giubergia), (14/04/2025)

    % Add the path to the folder containing the NORDIC code
    addpath('/home/matuser/matlab_toolbox/NORDIC_Raw-main/');
    addpath('/home//matuser/matlab_toolbox/NIFTI_toolbox/');


    [folder, fn, ext] = fileparts(output_file);
    if strcmp(ext, '.gz')
        [~, fn, ~] = fileparts(fn);
    end
    output_folder = fullfile(folder);
    nordic_fname = fn;
    
    % Create output folder if it does not exist
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    ARG.magnitude_only = 1;
    ARG.DIROUT = [ output_folder '/' ];
    ARG.write_gzipped_niftis = 1;

    % Identify real and imaginary files
    input_real = real_file;
    input_imag = imag_file;
    input_mag = mag_file;
    
    % make dir to host temporary files
    if ~exist(fullfile(output_folder, 'tmp'), 'dir')
        mkdir(fullfile(output_folder, 'tmp'));
    end    
    
    copyfile(input_mag, fullfile(output_folder, 'tmp', 'part-mag.nii.gz'));
    part_mag = fullfile(output_folder, 'tmp', 'part-mag.nii.gz');
    
    phase_output_file = '';
    if ~isempty(strip(input_imag)) && ~isempty(strip(input_real))
        copyfile(input_real, fullfile(output_folder, 'tmp','part-real.nii.gz'));
        part_real = fullfile(output_folder, 'tmp','part-real.nii.gz');
        copyfile(input_imag, fullfile(output_folder,'tmp', 'part-imag.nii.gz'));
        part_imag = fullfile(output_folder, 'tmp','part-imag.nii.gz');
        
        % Compute phase NIfTI file
        phase_output_file = fullfile(output_folder, 'tmp','part-phase.nii.gz');
        complex_output_file = fullfile(output_folder,'tmp', 'complex.nii.gz');
        bash_command = ['fslcomplex -complex ' part_real ' ' part_imag ' ' complex_output_file ];
        system(bash_command);
        bash_command = ['fslcomplex -realphase ' complex_output_file ' ' phase_output_file];
        system(bash_command);
        bash_command = ['fslcpgeom ' part_mag ' ' phase_output_file];
        system(bash_command);
        fprintf('Phase image computation completed. Saved as: %s\n', phase_output_file);
        ARG.magnitude_only = 0;
    end

    % Apply NORDIC denoising
    NIFTI_NORDIC(part_mag, phase_output_file, nordic_fname, ARG);

    % remove temporary folder
    rmdir(fullfile(output_folder, 'tmp'), 's');
    
    fprintf('Process completed. Denoised files saved in: %s\n', output_folder);
end


