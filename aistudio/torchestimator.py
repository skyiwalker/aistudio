import subprocess
import sys
import os
import pickle
import shutil
# for monitoring
from time import sleep
from lrcurve import PlotLearningCurve
# for get model
import torch

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
        
        ########## PATH ##########
        ##########################
        # TODO: Get workspace path
        ##########################
        WORKSPACE_PATH = '/home/sky/dev/aistudio/workspace/ws-1' 
        self.workspace_path = WORKSPACE_PATH
        ##########################
        # TODO: NEED JOB INDEX
        ##########################
        # Job index is a number after last job number
        JOB_INDEX = 1
        # Job path settings (mkdir)
        JOB_PATH = WORKSPACE_PATH + '/jobs/job-' + str(JOB_INDEX)
        self.job_path = JOB_PATH
        JOB_SCRIPT = JOB_PATH+'/run.sh'
        self.job_script = JOB_SCRIPT
        # Add Model Path
        if self.model_name is not "":
            MODEL_PATH = self.workspace_path + '/models/' + self.model_name
            args.append('--model-path')
            args.append(MODEL_PATH)
            self.MODEL_PATH = MODEL_PATH
            self.trained = True
        # Add Network Name
        elif self.net_name is not "":            
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
        
        with open(self.job_script,'w') as shfile:
            shell_script='''\
#!/bin/sh
horovodrun -np {} python {} {}           
'''.format(str(self.nprocs),self.job_path+'/train.py',argstr)
            shfile.write(shell_script)
        # Set permission to run the script
        os.chmod(self.job_script, 0o777)
        # Save Dataset
        if input_data is not None and input_labels is not None:
            with open(self.job_path+'/dataset.pkl', 'wb') as f:
                pickle.dump((input_data, input_labels), f, pickle.HIGHEST_PROTOCOL)
        # Request Job Submission
        self.trained = self._request_to_portal()
        
    def register_model(self, model_name):
        # add model information to database(file db or web db)
        # create model folder        
        model_path = self.workspace_path + '/models/' + model_name
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        # copy network file to model path
        # $WORKSPACE/nets/{net_name} -> $WORKSPACE/models/{model_name}/torchmodel.py
        org_net_path = self.workspace_path + '/nets/' + self.net_name + '.py'
        net_path = model_path + '/torchmodel.py'
        shutil.copy(org_net_path, net_path)
        # copy model file ti model path
        # $JOB_PATH/torchmodel.pth -> $WORKSPACE/models/{model_name}/torchmodel.pth
        org_modelfile_path = self.job_path + '/torchmodel.pth'
        modelfile_path = model_path + '/torchmodel.pth'
        shutil.copy(org_modelfile_path, modelfile_path)
        
    def monitor(self):        
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
                    'acc': { 'name': 'Accuracy', 'limit': [0, 1], 'scale': 'linear' }
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
        with open(self.job_path + "/epochs.log","r") as f:
            while True:
                where = f.tell()
                line = f.readline().strip()
                if not line:
                    sleep(0.1)
                    delay_time += 0.1
                    f.seek(where)
                    if delay_time > 10.0:
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
        
    def predict(self, dataset_loader=""):
        if dataset_loader is not "":
            if '--dataset-loader' not in self.args:
                self.args.append('--dataset-loader')
                self.args.append(dataset_loader)
        if '--dataset-loader' in self.args:
            argstr = ' '.join(self.args)
            argstr = argstr + ' --prediction'
            with open(self.job_script,'w') as shfile:
                shell_script='''\
#!/bin/sh
horovodrun -np {} python {} {}
'''.format(str(self.nprocs),self.job_path+'/train.py',argstr)
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
                # net path e.g.) $HOME/workspace/ws-1/nets
                netpath = os.path.join(self.workspace_path, 'nets')        
                sys.path.append(netpath)
                # Custom Model
                import importlib
                torchnet = importlib.import_module(modulename)
                # set model
                model = torchnet.Net()
                model.load_state_dict(torch.load(self.job_path+"/torchmodel.pth"))
                return model
        else:
            print("Training has not been completed.")
            return None
    
    def stop_job(self):
        _request_to_portal_stop_job()
        
    def _request_to_portal(self):
        print("Job Requested to Portal.")
        # lrcurve & pytorch crash
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        # TODO : Request to Portal for call slurm script
        proc = subprocess.Popen(self.job_script,
                                universal_newlines=True, # Good Expression for New Lines
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                )
        out = proc.communicate()[0]
        print(out)
        # if there is no error
        return True
    
    def _request_to_portal_stop_job(self):
        print("Stop a Requested Job on the Portal.")
    
    
        
        