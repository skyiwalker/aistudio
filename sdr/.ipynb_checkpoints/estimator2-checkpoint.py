import subprocess
import sys

class Estimator:
    def __init__(self,model=None,script_params={}):
        self.model = model
        self.script_params = script_params
        
    def fit(self,input_data,input_labels):
        sys.path.append('..')
        # use full train script
        exec_script = 'train_hvd.py'
        args = []
        for key, value in self.script_params.items():
            args.append(str(key))
            args.append(str(value))
        print(args)
        proc = subprocess.Popen(['horovodrun','-np','1','python',exec_script,*args],
                                universal_newlines=True, # Good Expression for New Lines
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                )
#         proc = subprocess.Popen(['python',exec_script,*args],
#                                 universal_newlines=True, # Good Expression for New Lines
#                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                                 )
        out = proc.communicate()[0]
        print(out)