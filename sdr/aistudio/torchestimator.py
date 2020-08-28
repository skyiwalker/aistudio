import subprocess
import sys
import os
import pickle
import shutil
# for handling json
import json
import math
# for monitoring
from time import sleep
from lrcurve import PlotLearningCurve
# for get model
import torch
# to make datetime
from datetime import datetime
import requests
import time

MAX_DELAY_TIME = 600.0 # seconds => 600.0s = 10mins

class TorchEstimator:
    def __init__(self,model_name="",net_name="",script_params={}):
        self.model_name = model_name
        self.net_name = net_name
        self.script_params = script_params
        self.init_params()
        self.apiurl = 'https://sdr.edison.re.kr:8443'
        
    def init_params(self):
        # Handling Arguments
        # Number of processors to run MPI
        nprocs = 1
        if 'nprocs' in self.script_params:
            nprocs = self.script_params['nprocs']
        self.nprocs = nprocs        
        if 'debug' in self.script_params:
            self.debug = True
        if 'validation' in self.script_params:
            self.validation = True
        else:
            self.validation = False
        # Make Arguments List
        args = []
        for key, value in self.script_params.items():
            if key == "no-cuda" and value == "False":
                pass
            if key == "debug" and value == "False":
                pass
            if key == "validation" and value == "False":
                pass
            else:
                args.append('--'+key) # names of arguments
            if key != 'debug' and key != 'no-cuda' and key != 'validation':                
                args.append(str(value))
        
        # Initialize trained param
        self.trained = False
        
        ########## LOAD EQUIPMENT SPECS ##########
        max_node = 4
        num_core_cpu = 32
        num_core_gpu = 2
        
        ########## PATH ##########
        ## Estimator's Directory #
        # Get path for this module
        self.this_path = os.path.dirname(os.path.abspath(__file__))
        ##########################
        # Get workspace path
        ##########################                
        EDISON_SCIDATA_PATH = '/EDISON/SCIDATA/sdr/draft'
        notebook_path = os.getcwd()
        print(notebook_path)
        pos1 = notebook_path.find('workspace')
        if pos1 == -1:
            print("Wrong workspace path")
            return
        pos2 = notebook_path[pos1+10:].find('/')
        WORKSPACE_PATH = ''
        if pos2 == -1:
            print("Workspace path was found.")
            WORKSPACE_PATH = notebook_path
            self.workspace_name = notebook_path[pos1+10:]
        else:
            print("Warning: Path of the notebook file must be located at the root of the workspace folder.")
            WORKSPACE_PATH = notebook_path[:pos1+10+pos2] # workspace is 9 letters   
            self.workspace_name = notebook_path[pos1+10:pos1+10+pos2]
        self.workspace_path = WORKSPACE_PATH
        # get user id from workspace path
        self.user_id = WORKSPACE_PATH[6:6+(WORKSPACE_PATH[6:].find('/'))]
        self.home_path = EDISON_SCIDATA_PATH + '/' + self.user_id
        # Job path settings (mkdir)
#         JOB_PATH = WORKSPACE_PATH + '/job/job-' + str(JOB_INDEX)        
        JOB_PATH = WORKSPACE_PATH + '/job'
        print(JOB_PATH)
        self.job_path = JOB_PATH
        self.has_job = False
        # 5 is after 'home/'        
        self.real_output_path = EDISON_SCIDATA_PATH + JOB_PATH[5:]
        self.real_workspace_path = EDISON_SCIDATA_PATH + WORKSPACE_PATH[5:]
        # Add Model Path
        if self.model_name is not "":
            MODEL_PATH = WORKSPACE_PATH + '/model/' + self.model_name
            args.append('--model-path')
            args.append(MODEL_PATH)
            self.script_params['model-path'] = MODEL_PATH
            self.model_path = MODEL_PATH
            self.trained = True
        # Add Network Name
        elif self.net_name is not "":            
            args.append('--net-name')
            args.append(self.net_name)
            self.script_params['net-name'] = self.net_name
        # Verify Arguments
        if self.debug:
            print(args)            
        self.args = args

    def make_job_path(self):
        timenow = datetime.now().strftime('%Y%m%d%H%M%S')
        self.job_title = 'job-' + timenow
        self.this_job_path = self.job_path + '/' + self.job_title        
        self.job_script = self.this_job_path + '/job.sh'
        self.output_path = self.real_output_path + '/' +  self.job_title
        if not os.path.isdir(self.this_job_path):
            os.mkdir(self.this_job_path)
        self.has_job = True
        return self.this_job_path
        
    def make_shell_script(self,argstr):                
        with open(self.this_path+"/eqspec.json", "r") as eqspec_json:
            eqspec = json.load(eqspec_json)
        num_cores_cpu = eqspec['num-cores-cpu']
        num_cores_gpu = eqspec['num-cores-gpu']
        max_nodes = eqspec['max-nodes']
        
        #calculate max procs
        max_procs_cpu = num_cores_cpu * max_nodes
        max_procs_gpu = num_cores_gpu * max_nodes
                
        if 'no-cuda' in self.script_params:
            # use cpu
            if self.nprocs > max_procs_cpu:
                print("The maximum number of cores has been exceeded.")
                return "ERROR"
        else:
            # use gpu
            if self.nprocs > max_procs_gpu:
                print("The maximum number of cores has been exceeded.")
                return "ERROR"
        
        # calculate ntasks and nnodes
        ntasks = self.nprocs
        nnodes = min(max_nodes,ntasks)
        ntasks_per_node = math.ceil(ntasks/nnodes)
        
        shell_script='''\
#!/bin/bash
#SBATCH --job-name=job-1
#SBATCH --output={}/std.out
#SBATCH --error={}/std.err
#SBATCH --nodes={}
#SBATCH --ntasks={}
#SBATCH --ntasks-per-node={}
#SBATCH --exclusive

HOME={}
JOBDIR={}
conda activate torch
/usr/local/bin/mpirun -np {} -x TORCH_HOME=/home/{} -x PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x NCCL_SOCKET_IFNAME=^docker0,lo -mca btl_tcp_if_exclude lo,docker0  -mca pml ob1 singularity exec --nv -H ${{HOME}}:/home/{} --nv --pwd ${{JOBDIR}} /EDISON/SCIDATA/singularity-images/userenv3 python ${{JOBDIR}}/train.py {}
curl {}/api/jsonws/SDR_base-portlet.dejob/studio-update-status -d deJobId={} -d Status=SUCCESS 
'''.format(self.output_path, self.output_path, nnodes, ntasks, ntasks_per_node, self.home_path, self.this_job_path, str(self.nprocs), self.user_id, self.user_id, argstr, self.apiurl, self.job_id)        
        
        # FOR LOCAL TEST
#         shell_script='''\
# #!/bin/sh
# horovodrun -np {} python {} {}
# '''.format(str(self.nprocs), self.this_job_path+'/train.py', argstr)
        
        return shell_script

    def get_job_path(self):
        if self.has_job:
            return self.this_job_path
        else:
            print("A working job directory has not been created yet.")
            
    def write_metadata(self):
        # Example of script_params
        '''
        script_params = {
            'epochs':5,
            'batch-size':64,
            'test-batch-size':128,
            'lr':0.01,
            'momentum':0.5,
            'seed':42,
            'log-interval':10,
            #'no-cuda':False,
            'nprocs':1,
            'loss':'cross_entropy',
            #'loss':'nll_loss',
            'optimizer':'SGD',
            'validation': True,
            'debug': True
        }
        '''
                
        metadata = {}
        metadata['hyperparameters'] = {}
        metadata['otherparameters'] = {}
        metadata['others'] = {}
        
        for key, value in self.script_params.items():
            if key == "epochs" or key == "batch-size" or key =="test-batch-size" or \
                key == "lr" or key == "momentum" or key == "loss" or key == "optimizer":
                metadata['hyperparameters'][key] = value
            elif key == "seed" or key =="log-interval" or key =="nprocs" or key =="validation" or \
                key =="debug" or key =="net-name" or key =="dataset-loader" or key =="no-cuda":
                metadata['otherparameters'][key] = value
            else:
                metadata['others'][key] = value

        with open(self.this_job_path + "/meta-job.json", "w") as json_file:
            json.dump(metadata, json_file)
        
    def copy_train_script(self):
        if self.has_job:
            # copy train.py to job path
            org_train_script_path = self.this_path + '/train.py'
            train_script_path = self.this_job_path + '/train.py'
            shutil.copy(org_train_script_path, train_script_path)
    
    def fit(self,input_data=None,input_labels=None,dataset_loader=""):
        # Add Dataset Loader Name to Arguments
        if dataset_loader is not "":
            self.args.append('--dataset-loader')
            self.args.append(dataset_loader)
            self.script_params['dataset-loader'] = dataset_loader
        # make arguments list to one string
        argstr = ' '.join(self.args)        
        ##### Make dir for new job #####
        self.make_job_path()
        ##### request submit job (register job to database) - API Call #####
        self._request_submit_job()
        # copy train.py to job path
        self.copy_train_script()
        # Writing Training Script        
        with open(self.job_script,'w') as shfile:
            ##### Make Shell Script #####
            shell_script=self.make_shell_script(argstr)
            if shell_script == "ERROR":
                # Failed to Predict
                return
            shfile.write(shell_script)
        # Set permission to run the script
        os.chmod(self.job_script, 0o777)
        # Save Dataset
        if input_data is not None and input_labels is not None:
            with open(self.this_job_path+'/dataset.pkl', 'wb') as f:
                pickle.dump((input_data, input_labels), f, pickle.HIGHEST_PROTOCOL)
        ##### Write Meta Data JSON File #####
        self.write_metadata()
        # Request Job Submission
        self.trained = self._request_to_portal()
        
    def register_model(self, model_name):
        # add model information to database(file db or web db)
        # create model folder        
        model_path = self.workspace_path + '/model/' + model_name
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        # copy network file to model path
        # $WORKSPACE/nets{net_name} -> $WORKSPACE/model/{model_name}/torchmodel.py
        org_net_path = self.workspace_path + '/net/' + self.net_name + '.py'
        net_path = model_path + '/torchmodel.py'
        shutil.copy(org_net_path, net_path)
        # copy model file ti model path
        # $JOB_PATH/torchmodel.pth -> $WORKSPACE/model/{model_name}/torchmodel.pth
        org_modelfile_path = self.this_job_path + '/torchmodel.pth'
        modelfile_path = model_path + '/torchmodel.pth'
        shutil.copy(org_modelfile_path, modelfile_path)
        
    def monitor(self, timeout=MAX_DELAY_TIME):        
        if 'epochs' in self.script_params:
            self.epochs = self.script_params['epochs']
        # for end condition
        epochs = int(self.epochs)
        
        if self.validation:
            plot = PlotLearningCurve(
                mappings = {
                    'loss': { 'line': 'train', 'facet': 'loss' },
                    'val_loss': { 'line': 'validation', 'facet': 'loss' },
                    'acc': { 'line': 'train', 'facet': 'acc' },
                    'val_acc': { 'line': 'validation', 'facet': 'acc' }
                },
                facet_config = {
                    'loss': { 'name': 'Loss', 'limit': [None, None], 'scale': 'linear' },
                    'acc': { 'name': 'Accuracy', 'limit': [None, 1], 'scale': 'linear' }
                },
                xaxis_config = { 'name': 'Epoch', 'limit': [0, epochs] }
            )
        else:
            plot = PlotLearningCurve(
                mappings = {
                    'loss': { 'line': 'train', 'facet': 'loss' },
                    'acc': { 'line': 'train', 'facet': 'acc' }
                },
                facet_config = {
                    'loss': { 'name': 'Loss', 'limit': [None, None], 'scale': 'linear' },
                    'acc': { 'name': 'Accuracy', 'limit': [0, 1], 'scale': 'linear' }
                },
                xaxis_config = { 'name': 'Epoch', 'limit': [0, epochs] }
            )
        # log monitoring loop
        delay_time = 0.
        try:
            with open(self.this_job_path + "/epoch.log","r") as f:
                while True:
                    where = f.tell()
                    line = f.readline().strip()
                    if not line:
                        sleep(0.1)
                        delay_time += 0.1
                        f.seek(where)
                        if timeout is not None:
                            if delay_time > timeout:
                                print("Delay has been exceeded.")
                                break
                        else:
                            if delay_time > MAX_DELAY_TIME:
                                print("Delay has been exceeded.")
                                break                    
                    else:
                        delay_time = 0. # reset delay time
                        # print(line) # already has newline
                        strlist = line.split(',')
                        phase = strlist[0].strip()
                        epoch = strlist[1].strip()
                        loss = strlist[2].strip()
                        acc = strlist[3].strip()
                        print("Phase: {}, Epoch: {}, Loss: {}, Acc: {}".format(phase,epoch,loss,acc))
                        # append and update
                        if phase=="Train":
                            plot.append(epoch, {
                                'loss': loss,
                                'acc': acc
                            })
                            plot.draw()
                        else:
                            plot.append(epoch, {
                                'val_loss': loss,
                                'val_acc': acc
                            })
                            plot.draw()
                        # End Condition (If the Last Epoch Finished, Terminate the Loop)
                        if self.validation:
                            if phase=="Validation":
                                if int(epoch) == epochs:
                                    break
                        else:
                            if int(epoch) == epochs:
                                break 
        except:
            print("Even one epoch has not been completed.\nPlease execute the command again after a while.")        
                        
        
    def predict(self, dataset_loader=""):
        if dataset_loader is not "":
            if '--dataset-loader' not in self.args:
                self.args.append('--dataset-loader')
                self.args.append(dataset_loader)
                self.script_params['dataset-loader'] = dataset_loader
        if '--dataset-loader' in self.args:
            argstr = ' '.join(self.args)
            argstr = argstr + ' --prediction'
            if self.has_job==False:
                ##### Make dir for new job #####
                self.make_job_path()                
                # copy train.py to job path
                self.copy_train_script()
            ##### request submit job (register job to database) - API Call #####
            self._request_submit_job()
            # FOR LOCAL TEST
            with open(self.job_script,'w') as shfile:
#                 shell_script='''\
# #!/bin/sh
# horovodrun -np {} python {} {}
# '''.format(str(self.nprocs),self.job_path+'/train.py',argstr)
                ##### Make Shell Script #####
                shell_script=self.make_shell_script(argstr)
                if shell_script == "ERROR":
                    # Failed to Predict
                    return
                shfile.write(shell_script)
            # Set permission to run the script
            os.chmod(self.job_script, 0o777)
            ##### Write Meta Data JSON File #####
            self.write_metadata()
            # Request Job Submission
            self.trained = self._request_to_portal()
        else:
            print("Dataset Loader Not Found.")
            return
    
    # Report for Prediction
    def report(self):
        try:
            with open(self.this_job_path + "/epoch_prediction.log","r") as f:
                line = f.readline()
                print(line)
        except:
            print("Prediction has not been completed.\nPlease execute the command again after a while.")        
        
    def get_model(self):
        # Load Network
        if self.trained:
            if self.net_name is not "":
                print("Network was found.")
                # set system path to load model
                modulename = self.net_name
                # net path e.g.) $HOME/workspace/ws-1/net
                netpath = os.path.join(self.workspace_path, 'net')        
                sys.path.append(netpath)
                # Custom Model
                import importlib
                torchnet = importlib.import_module(modulename)
                # set model
                model = torchnet.Net()
                model.load_state_dict(torch.load(self.this_job_path+"/torchmodel.pth"))
                return model
        else:
            print("Training has not been completed.")
            return None
    
    def _request_submit_job(self):
        data = {
          'screenName': self.user_id,
          'title': self.job_title,
          'targetType': '81', # targetType 81 is for normal ai job(train,predict)
          'workspaceName': self.workspace_name,
          'location': self.real_workspace_path #self.workspace_path
        }
        if self.debug:
            print(data)        
        response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/studio-submit-de-job', data=data)
        if response.status_code == 200:            
            self.job_id = response.json()            
            print("Job ID-{} was submitted.".format(self.job_id))
        else:
            print("A problem occured when generating the job.")
        print("Job was generated in database.")
    
    def _run_slurm_script(self):
        data = {
          'screenName': self.user_id,
          'location': self.output_path
        }
        if self.debug:
            print(data)
        print("Running Slurm script...")
        response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/slurm-de-job-run', data=data)
        # waiting for slurm job id
        time.sleep(3)
        try:
            with open(self.this_job_path+'/job.id','r') as f:
                idstr = f.readline()
                idstr = idstr.strip()
                print("Batch job ID-{} is running on the HPC.".format(idstr))
                self.slurm_job_id = int(idstr)
                self._request_update_status("RUNNING")
        except:
            print("The requested training job has failed.")
        
    def status(self):
        self._request_get_status()
        
    def _request_update_status(self,status):        
        try:
            data = {
              'deJobId': self.job_id,
              'Status': status
            }
            response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/studio-update-status', data=data)
            if self.debug:
                if response.status_code == 200:
                    print("Job status({}) has been updated.".format(status))
        except:
            print("Error: Slurm Job Not Found.")
        
        
    def _request_get_status(self):
        print("Getting Status of Requested Job on the Portal.")
        try:
            data = {
              'deJobId': self.job_id
            }
            response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/get-de-job-data', data=data)
            if response.status_code == 200:
                resjson = response.json()
                print('--------------------------------')
                print('Job ID: {}'.format(resjson['deJobId']))
                print('Job Title: {}'.format(resjson['title']))
                print('Start Date: {}'.format(resjson['startDt']))
                print('End Date: {}'.format(resjson['endDt']))
                print('--------------------------------')
                print('Status: {}'.format(resjson['status']))
            else:
                print("Error: Getting status of the job has failed.")
        except:
            print("Error: Running Job Not Found.")
    
    def cancel(self):
        self._request_to_portal_cancel_job()
        
    def _request_to_portal(self):
        print("Job Requested to Portal.")
        # FOR LOCAL TEST
        # lrcurve & pytorch crash
#         os.environ['MKL_THREADING_LAYER'] = 'GNU'        
#         proc = subprocess.Popen(self.job_script,
#                                 universal_newlines=True, # Good Expression for New Lines                                
#                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                                 )
#         try:
#             outs, errs = proc.communicate(timeout=15)
#         except subprocess.TimeoutExpired:
#             proc.kill()
#             outs, errs = proc.communicate()
#         print(outs)        
        # Request to Portal to call slurm script
        self._run_slurm_script()
        # if there is no error
        return True
    
    def _request_to_portal_cancel_job(self):
        print("Canceling a Requested Job on the Portal.")
        try:
            data = {
              'jobId': self.slurm_job_id,
              'screenName': self.user_id
            }
            response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/slurm-de-job-cancel', data=data)
            if response.status_code == 200:
                print("The job was successfully canceled.")
                self._request_update_status("CANCELLED")
        except:
            print("Error: Slurm Job Not Found.")