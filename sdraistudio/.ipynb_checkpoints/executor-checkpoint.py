import subprocess
import sys
import pickle
import os

class Executor:
    def __init__(self,model=None,script_params={}):
        self.model = model
        self.script_params = script_params
        
    def fit(self,input_data,input_labels):
        sys.path.append('..')        
        # save data
        # save
        PATH = "./train-data-job-1/"
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        with open(PATH+'data-labels.pkl', 'wb') as f:
            pickle.dump((input_data, input_labels), f, pickle.HIGHEST_PROTOCOL)        
        # make module
        code_str = '''
import pickle

def load_data():
    # load
    with open('{}', 'rb') as f:
        data = pickle.load(f)
    return data
'''.format(PATH+'data-labels.pkl')
        
        print(code_str)
        with open('./dataset1.py', 'w') as mod:
            mod.write(code_str)
        
        # use full train script
        exec_script = 'train_some.py'
        args = []
        for key, value in self.script_params.items():
            args.append(str(key))
            args.append(str(value))
        print("arguments: ", args)
        proc = subprocess.Popen(['python',exec_script,*args],
                                universal_newlines=True, # Good Expression for New Lines
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                )
        out = proc.communicate()[0]
        print(out)