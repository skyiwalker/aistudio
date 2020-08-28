import subprocess
import sys
import os
import pickle
import shutil

class Estimator:
    def __init__(self,model_name="",net_name="",script_params={}):
        self.model_name = model_name
        self.net_name = net_name
        self.script_params = script_params
        self.init_params()
        
    def init_params(self):
        # Handling Arguments
        # Number of processors to run MPI
        nprocs = 1
        if 'nprocs' in self.script_params:
            nprocs = self.script_params['nprocs']
        self.nprocs = nprocs
        debug = False
        if 'debug' in self.script_params:
            debug = self.script_params['debug']
        # Make Arguments List
        args = []
        for key, value in self.script_params.items():
            args.append('--'+key) # names of arguments            
            if key != 'debug' and key != 'no-cuda' and key != 'validation':                
                args.append(str(value))
        
        ##########################
        # TODO: Get workspace path
        ##########################
        self.WORKSPACE_PATH = '/home/sky/dev/aistudio/workspace/ws-1'        
        # Add Model Path
        if self.model_name is not "":
            MODEL_PATH = self.WORKSPACE_PATH + '/models/' + self.model_name
            args.append('--model-path')
            args.append(MODEL_PATH)
            self.MODEL_PATH = MODEL_PATH
        # Add Network Name
        if self.net_name is not "":            
            args.append('--net-name')
            args.append(self.net_name)
        # Verify Arguments
        if debug:
            print(args)
        self.args = args
        
    def fit(self,input_data=None,input_labels=None,dataset_loader=""):
        # Add Dataset Loader Name to Arguments
        if dataset_loader is not "":
            self.args.append('--dataset-loader')
            self.args.append(dataset_loader)
        # make arguments list to one string
        argstr = ' '.join(self.args)        
        # Writing Training Script
        ##########################
        # TODO: NEED JOB INDEX
        ##########################
        # Job index is a number after last job number
        JOB_INDEX = 1
        # Job path settings (mkdir)
        JOB_PATH = self.WORKSPACE_PATH+'/jobs/job-'+str(JOB_INDEX)+'/'
        self.JOB_PATH = JOB_PATH
        JOB_SCRIPT = JOB_PATH+'run.sh'
        self.job_script = JOB_SCRIPT
        with open(JOB_SCRIPT,'w') as shfile:
            shell_script='''\
#!/bin/sh
horovodrun -np {} python {} {}           
'''.format(str(self.nprocs),JOB_PATH+'train.py',argstr)
            shfile.write(shell_script)
        # Set permission to run the script
        os.chmod(JOB_SCRIPT, 0o777)        
        # Save Dataset
        if input_data is not None and input_labels is not None:
            with open(JOB_PATH+'dataset.pkl', 'wb') as f:
                pickle.dump((input_data, input_labels), f, pickle.HIGHEST_PROTOCOL)
        # Request Job Submission
        self._request_to_portal()
        
    def register_model(self, model_name):
        # add model information to database(file db or web db)
        # create model folder        
        model_path = self.WORKSPACE_PATH + '/models/' + model_name
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        # copy network file to model path
        # $WORKSPACE/nets/{net_name} -> $WORKSPACE/models/{model_name}/torchmodel.py
        org_net_path = self.WORKSPACE_PATH + '/nets/' + self.net_name + '.py'
        net_path = model_path + '/torchmodel.py'
        shutil.copy(org_net_path, net_path)
        # copy model file ti model path
        # $JOB_PATH/torchmodel.pth -> $WORKSPACE/models/{model_name}/torchmodel.pth
        org_modelfile_path = self.JOB_PATH + 'torchmodel.pth'
        modelfile_path = model_path + '/torchmodel.pth'
        shutil.copy(org_modelfile_path, modelfile_path)
        
    def _request_to_portal(self):
        print("Job Requested to Portal.")
        # TODO : Request to Portal for call slurm script
        proc = subprocess.Popen(self.job_script,
                                universal_newlines=True, # Good Expression for New Lines
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                )
        out = proc.communicate()[0]
        print(out)