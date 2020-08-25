import subprocess

class PyTorch:
    def __init__(self,script,script_params={},directory="",use_gpu=False,nprocs=1):
        self.script = script
        self.script_params = script_params
        self.directory = directory
        self.use_gpu = use_gpu
        self.nprocs = nprocs
    def fit(self):
        exec_script = self.script
        args = []
        for key, value in self.script_params.items():
            args.append(str(key))
            args.append(str(value))
        print(args)
        proc = subprocess.Popen(['python',exec_script,*args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = proc.communicate()
        print(out)