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

MAX_DELAY_TIME = 600.0 # seconds => 600.0s = 10mins

class TorchEstimator:
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
        if 'validation' in self.script_params:
            self.validation = True
        else:
            self.validation = False
        # Make Arguments List
        args = []
        for key, value in self.script_params.items():
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
        # TODO: Get workspace path
        ##########################
        WORKSPACE_PATH = '/home/sky/dev/aistudio/workspace/ws-1'
        EDISON_SCIDATA_PATH = '/EDISON/SCIDATA/sdr/draft'
        self.workspace_path = WORKSPACE_PATH
        # get user id from workspace path
        self.user_id = WORKSPACE_PATH[6:6+(WORKSPACE_PATH[6:].find('/'))]
        self.home_path = EDISON_SCIDATA_PATH + '/' + self.user_id
        ##########################
        # TODO: NEED JOB INDEX
        ##########################
        # Job index is a number after last job number
        #JOB_INDEX = 1
        # Job path settings (mkdir)
#         JOB_PATH = WORKSPACE_PATH + '/job/job-' + str(JOB_INDEX)        
        JOB_PATH = WORKSPACE_PATH + '/job'
        self.job_path = JOB_PATH
        # 5 is after 'home/'
        OUTPUT_PATH = EDISON_SCIDATA_PATH + JOB_PATH[5:]
        self.output_path = OUTPUT_PATH
        # Add Model Path
        if self.model_name is not "":
            MODEL_PATH = WORKSPACE_PATH + '/model/' + self.model_name
            args.append('--model-path')
            args.append(MODEL_PATH)
            self.model_path = MODEL_PATH
            self.trained = True
        # Add Network Name
        elif self.net_name is not "":            
            args.append('--net-name')
            args.append(self.net_name)
        # Verify Arguments
        if debug:
            print(args)
        self.args = args

    def make_job_path(self):
        timenow = datetime.now().strftime('%Y%m%d%H%M%S')
        new_job_path = self.job_path + '/job-' + timenow
        self.this_job_path = new_job_path
        self.job_script = new_job_path + '/run.sh'
        self.output_path = self.output_path + '/job-' + timenow
        if not os.path.isdir(new_job_path):
            os.mkdir(new_job_path)
        return new_job_path
        
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
/usr/local/bin/mpirun -np {} -x PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x NCCL_SOCKET_IFNAME=^docker0,lo -mca btl_tcp_if_exclude lo,docker0  -mca pml ob1 singularity exec --nv -H ${{HOME}}:/home/{} --nv --pwd ${{JOBDIR}} /EDISON/SCIDATA/singularity-images/userenv3 python ${{JOBDIR}}/train.py {}
'''.format(self.output_path, self.output_path, nnodes, ntasks, ntasks_per_node, self.home_path, self.this_job_path, str(self.nprocs), self.user_id, argstr)
        
        # FOR LOCAL TEST
        shell_script='''\
horovodrun -np {} python ./train.py {}
'''.format(str(self.nprocs), argstr)
        
        return shell_script
        
    def fit(self,input_data=None,input_labels=None,dataset_loader=""):
        # Add Dataset Loader Name to Arguments
        if dataset_loader is not "":
            self.args.append('--dataset-loader')
            self.args.append(dataset_loader)
        # make arguments list to one string
        argstr = ' '.join(self.args)
        # mkdir for new job
        self.make_job_path()
        # copy train.py to job path
        org_train_script_path = self.this_path + '/train.py'
        train_script_path = self.this_job_path + '/train.py'
        shutil.copy(org_train_script_path, train_script_path)
        # Writing Training Script        
        with open(self.job_script,'w') as shfile:
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
        # for end condition
        epochs = self.script_params['epochs']
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
        if '--dataset-loader' in self.args:
            argstr = ' '.join(self.args)
            argstr = argstr + ' --prediction'
            with open(self.job_script,'w') as shfile:
#                 shell_script='''\
# #!/bin/sh
# horovodrun -np {} python {} {}
# '''.format(str(self.nprocs),self.job_path+'/train.py',argstr)
                shell_script=self.make_shell_script(argstr)
                if shell_script == "ERROR":
                    # Failed to Predict
                    return
                shfile.write(shell_script)
            # Set permission to run the script
            os.chmod(self.job_script, 0o777)
            # Request Job Submission
            self.trained = self._request_to_portal()
        else:
            print("Dataset Loader Not Found.")
            return
        
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
    
    def stop(self):
        _request_to_portal_stop_job()
        
    def _request_to_portal(self):
        print("Job Requested to Portal.")
        # lrcurve & pytorch crash
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        # TODO : Request to Portal for call slurm script
#         proc = subprocess.Popen(self.job_script,
#                                 universal_newlines=True, # Good Expression for New Lines
#                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                                 )
#         out = proc.communicate()[0]
#         print(out)
        # if there is no error
        return True
    
    def _request_to_portal_stop_job(self):
        print("Stop a Requested Job on the Portal.")
    
    
        
        