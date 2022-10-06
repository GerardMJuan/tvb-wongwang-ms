__author__ = 'gsanroma'

import os
import subprocess
from time import sleep
from sys import platform

#
# # Class sbatch launcher
#

class Launcher(object):

    def __init__(self, cmd):

        self.name = 'script.sh'
        self.folder = './'
        self.queue = 'high'
        self.cmd = cmd
        self.mem = 12000
        self.sockets = 1
        self.cores = 10
        self.threads = 1
        self.omp_num_threads = 2

        if platform == 'darwin':
            self.is_hpc = False
        else:
            self.is_hpc = True

    def run(self):

        script_file = os.path.join(self.folder, self.name + '.sh')
        f = open(script_file,'w')


        outfile = "{}.out".format(os.path.join(self.folder,self.name))
        errfile = "{}.err".format(os.path.join(self.folder,self.name))
        try:
            os.remove(outfile)
            os.remove(errfile)
        except:
            pass
        executable = "sbatch"
        f.write("#!/bin/bash\n")
        if self.name[0].isdigit():
            scriptname = 's' + self.name
        else:
            scriptname = self.name

        f.write("#SBATCH -J {}\n".format(scriptname))
        f.write("#SBATCH -p {}\n".format(self.queue))

        # hard  coded
        f.write("#SBATCH -N 1\n")
        f.write("#SBATCH --sockets-per-node {}\n".format(self.sockets))
        f.write("#SBATCH --cores-per-socket {}\n".format(self.cores))
        f.write("#SBATCH --threads-per-core {}\n".format(self.threads))

        f.write("#SBATCH -o {}\n".format(outfile))
        f.write("#SBATCH -e {}\n".format(errfile))

        f.write("#SBATCH --export=OMP_NUM_THREADS=1\n")
        f.write("source /etc/profile.d/lmod.sh\n")

        f.write("module load GCC\n")
        f.write("module load GSL\n")

        f.write('export PATH="$HOME/project/anaconda3/bin:$PATH"\n')
        f.write("source activate tvb\n")

        f.write(self.cmd)
        f.close()

        ex_out = subprocess.check_output([executable,script_file])
        return ex_out.split()[3]


def check_file_repeat(filename,n_repeats=5,wait_seconds=5):

    i = 0
    f = None
    while i < n_repeats:
        try:
            f = open(filename,'r')
        except:
            i += 1
            print("Failed attempt at reading {}".format(filename))
            sleep(wait_seconds)
            print("Retrying...")
            continue
        break
    assert f, "Cannot open qsub output file: {}".format(filename)
