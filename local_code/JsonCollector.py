import time
import os
import paramiko
import sys

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
        self.experience = experience
        self.remoteFolder = f"/home/yourusername/autodemo/irace/{experience}"
        self.JsonPath = self.experience.rsplit("_", 1)[0]
        os.makedirs(f"./JSON/{self.JsonPath}", exist_ok=True)
 
    def sshGet(self):
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
                remotePath = f"{self.remoteFolder}/{self.experience}.json"
                localPath = f"./JSON/{self.JsonPath}/{self.experience}.json"
                sftp.get(remotePath, localPath)
                sftp.close()
                ssh.close()
                failed = False
            except:
                print(f"ssh get failed")
                time.sleep(30)

if __name__ == '__main__':
    experience = sys.argv[1]
    number = sys.argv[2] 
    for i in range(int(number)):
        autodemo = AUTODEMO(f"{experience}_{i+1}")
        autodemo.sshGet()

    
    