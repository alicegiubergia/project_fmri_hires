function Bias_field_spm(input_file)

    addpath('/home//matuser/matlab_toolbox/spm12/');
    
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {input_file};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.0001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 20;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];
    matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];
    
    spm('defaults','FMRI')
    spm_jobman('initcfg');
    spm_jobman('run',matlabbatch);
end