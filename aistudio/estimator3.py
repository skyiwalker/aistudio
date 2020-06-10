import subprocess
import sys
import os
import pickle

class Estimator:
    def __init__(self,model=None,modelname="",script_params={}):
        self.model = model
        self.modelname = modelname
        self.script_params = script_params
        
    def fit(self,input_data,input_labels):
        # Handling Arguments
        # Number of processors to run MPI
        nprocs = 1
        if 'nprocs' in self.script_params:
            nprocs = self.script_params['nprocs']
        debug = False
        if 'debug' in self.script_params:
            debug = self.script_params['debug']        
        # Make Arguments List
        args = []
        for key, value in self.script_params.items():
            args.append('--'+str(key)) # names of arguments
            args.append(str(value))
        # TODO: Get workspace path
        WORKSPACE_PATH = '/home/sky/dev/aistudio'
        # Add Model Path
        MODEL_PATH = WORKSPACE_PATH + '/models/' + self.modelname
        args.append('--model-path')
        args.append(MODEL_PATH)
        # Verify Arguments
        if debug:
            print(args)
        self.args = args
        # make arguments list to one string
        argstr = ' '.join(args)        
        # Writing Training Script       
        # TODO: NEED JOB INDEX
        JOB_INDEX = 1   
        JOB_PATH = WORKSPACE_PATH+'/jobs/job-'+str(JOB_INDEX)+'/'
        JOB_SCRIPT = JOB_PATH+'run.sh'
        self.job_script = JOB_SCRIPT
        with open(JOB_SCRIPT,'w') as shfile:
            shell_script='''\
#!/bin/sh
horovodrun -np {} python {} {}           
'''.format(str(nprocs),JOB_PATH+'train.py',argstr)
            shfile.write(shell_script)
        # Set permission to run the script
        os.chmod(JOB_SCRIPT, 0o777)
        # Save Dataset
        with open(JOB_PATH+'dataset.pkl', 'wb') as f:
            pickle.dump((input_data, input_labels), f, pickle.HIGHEST_PROTOCOL)       
        
        self._request_to_portal()
        
    def _request_to_portal(self):
        print("Job Requested to Portal.")
        # TODO : Request to Portal for call slurm script
        proc = subprocess.Popen(self.job_script,
                                universal_newlines=True, # Good Expression for New Lines
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                )
        out = proc.communicate()[0]
        print(out)