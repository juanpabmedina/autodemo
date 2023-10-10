from cmath import cos, exp, inf
import numpy as np
from numpy import linalg as LA
import random
import time
import os
import ast
import paramiko
import sys
import json
from datetime import datetime
from xml.dom import minidom
from math import exp, sqrt, sin, cos
from sklearn import svm

# from automode.architecture.finite_state_machine import State, Transition, FSM
# from automode.modules.chocolate import Behavior, Condition

"""
Features vectors phi is dim=40 and each features is relative to the distance from the center of white/black floor patches

ALGO:

input: User demos, eps
1) Generate 1 random initial PFSM P_0 and compute mu_0, i=1
2) Compute t_i and w_i with SVM or projection algorithm
3) if t_i < eps, return PFSM
4) Run irace with R_i = w_i . phi_i to compute new PFSM P_i
5) Compute mu_i = mu(P_i)
6) i = i+1 and go back to 2)
"""

class AUTODEMO:

    def __init__(self, experience):
        self.mission = experience.split("_")[0]
        self.remoteFolder = f"/home/yourusername/autodemo/irace/{experience}"
        self.sshCreateFolder()
        self.sshCopyFolder()

        self.simFile = f"./argosFiles/{self.mission}.argos"
        self.baseFile = f"./baseImg/base_{self.mission}.png"
        self.demoFile = f"./demoImg/{self.mission}/"
        
        remoteSimpath =  f"{self.remoteFolder}/mission-folder/{self.mission}.argos"
        remoteBasepath =  f"{self.remoteFolder}/mission-folder/base/base_{self.mission}.png"
        
    
        self.sshPut(self.simFile, remoteSimpath)
        self.sshPut(self.baseFile, remoteBasepath)
        for item in os.listdir(self.demoFile):

            self.demoFile = f"./demoImg/{self.mission}/{item}"
            remoteDemopath =  f"{self.remoteFolder}/mission-folder/demos/{item}"

            self.sshPut(self.demoFile, remoteDemopath)

    def sshPut(self, localpath, remotepath):
        failed = True
        while(failed):
            try:
                host = "your_host_cluster"
                password = "password"  
                username = "yourusername"
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname = host,username = username, password = password)
                sftp = ssh.open_sftp()
                sftp.put(localpath, remotepath)
                sftp.close()
                ssh.close()
                failed = False
            except:
                print(f"Failed to ssh put: \"{localpath}\"")
                time.sleep(1)

    def sshCreateFolder(self):
        failed = True
        while(failed):
            try:
                host = "your_host_cluster"
                password = "password"  
                username = "yourusername"
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname = host,username = username, password = password)
                sftp = ssh.open_sftp()
                stdin, stdout, stderr = ssh.exec_command(f"mkdir {self.remoteFolder}")
                sftp.close()
                ssh.close()
                failed = False
            except:
                print(f"SSH create folder failed: {stderr}")
                time.sleep(1)

    def sshCopyFolder(self):
        failed = True
        while(failed):
            try:
                host = "your_host_cluster"
                password = "password"  
                username = "yourusername"
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname = host,username = username, password = password)
                sftp = ssh.open_sftp()
                stdin, stdout, stderr = ssh.exec_command(f"cp -r /home/yourusername/autodemo/irace/example/* {self.remoteFolder}")
                sftp.close()
                ssh.close()
                if(len(stderr.readlines()) > 0):
                    failed = True
                else:
                    failed = False
            except:
                print(f"SSH copy folder failed")
                time.sleep(1)
        return stdout

    def sshCmd(self, cmd):
        failed = True
        while(failed):
            try:
                host = "your_host_cluster"
                password = "password"  
                username = "yourusername"
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname = host,username = username, password = password)
                sftp = ssh.open_sftp()
                stdin, stdout, stderr = ssh.exec_command(cmd)
                sftp.close()
                ssh.close()
                if(len(stderr.readlines()) > 0):
                    failed = True
                else:
                    failed = False
            except:
                print(f"SSH cmd: \"{cmd}\" failed")
                time.sleep(1)
        return stdout
        
if __name__ == '__main__':
    experience = sys.argv[1]
    number = sys.argv[2]
    iterations = sys.argv[3]
    for i in range(int(number)):
        autodemo = AUTODEMO(f"{experience}_{i+1}")
        autodemo.sshCmd(f"sbatch autodemo.slurm {experience}_{i+1} {iterations}")
        print(f"Launched iteration {i+1}")
        time.sleep(5)