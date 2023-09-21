import subprocess
from cmath import cos, exp, inf
import numpy as np
from numpy import linalg as LA
import os
import ast
import sys
import json
import cv2
from xml.dom import minidom
from math import exp, sqrt, sin, cos
from sklearn import svm

from automode.architecture.finite_state_machine import State, Transition, FSM
from automode.modules.chocolate import Behavior, Condition

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

    def __init__(self, experience, iteration):
        self.iteration = iteration
        self.experience = experience
        self.mission = self.experience.split("_")[0]
        self.folder = f"/home/jpmedina/autodemo/irace/{experience}"
        self.demoFile = f"{self.folder}/mission-folder/{self.mission}.argos"
        self.arenaD = 3
        # self.patches, self.obstacles = self.retrievePatches(self.demoFile)

        self.img_path_names = os.listdir(f"{self.folder}/mission-folder/demos/")
        self.img_base_path = f"{self.folder}/mission-folder/base/base_{self.mission}.png"
        self.img_base = cv2.imread(self.img_base_path)
        self.d_kernel = 100
        self.method = cv2.HISTCMP_BHATTACHARYYA

        # parse mu history for svm from muHistory.txt
        self.muHistory = []
        self.labelHistory = []
        with open(f"{self.folder}/muHistory", "r") as f:
            lines = f.readlines()
            for line in lines:
                self.muHistory.append(ast.literal_eval(line.split(";")[0]))
                self.labelHistory.append((int(line.split(";")[1])))
        self.muE = self.muHistory[0]

    
    def computeMu(self, exp=10):
        phi_list = []
        for _ in range(exp):
            phi = self.computePhi()
            print(f"phi: {phi}")
            phi_list.append(phi)

        mu = []
        for j in range(len(phi_list[0])):
            avg = 0
            for i in range(len(phi_list)):
                avg += phi_list[i][j]
            mu.append(avg/len(phi_list))
        
        print(f"mu = {mu}")

        return mu, phi_list

    def computePhi(self):
        """From the computed current PFSM,
        compute the features expectations mu
        in form of a list of position of all robots"""
        pfsm=" ".join(self.PFSM.convert_to_commandline_args())

        self.run_Argos(pfsm=pfsm, sleep=3)
        # extract robots position  from argos output file
        # computes features expectation with demo positions

        swarm_pos = []
        with open(f"{self.folder}/mission-folder/pos.mu", "r") as f:
            for pos in f.readlines():
                swarm_pos.append(ast.literal_eval("[" + pos[:-2] + "]"))
            print(f"swarm pos: {swarm_pos}")

        os.remove(f"{self.folder}/mission-folder/pos.mu")

        generate_img = self.img_generator(swarm_pos)

        _, phi = self.histogram_comparision(generate_img, self.img_base, d_kernel=self.d_kernel, stride=self.d_kernel+1, method=self.method)
        
        return phi
    
    def histogram_comparision(self, img1, img2, d_kernel, stride, method):

        kernel = np.ones((d_kernel,d_kernel), np.float32) / d_kernel**2

        len_img_y, len_img_x, _ = img1.shape
         
        len_k = len(kernel)
        

        if stride != 0 :
            len_out_y = int(len_img_y/(stride))
            len_out_x = int(len_img_x/(stride))
        else:
            len_out_y = int(len_img_y/(len_k)) 
            len_out_x = int(len_img_x/(len_k)) 

        #print(len_out_x, len_out_y)
        
        histogram = np.zeros((len_out_y,len_out_x))

        a = 0
        b = 0 

        for n in range(0, len_out_x+1):
            for m in range(0, len_out_y+1):

                a = m*stride
                b = n*stride

                img_c1 = img1[a:a+d_kernel,b:b+d_kernel]
                img_c2 = img2[a:a+d_kernel,b:b+d_kernel]

                hist1 = cv2.calcHist([np.uint8(img_c1)],[0,1,2],None,[8,8,8],[0,256, 0, 256, 0, 256])
                hist1 = cv2.normalize(hist1, hist1).flatten()

                hist2 = cv2.calcHist([np.uint8(img_c2)],[0,1,2],None,[8,8,8],[0,256, 0, 256, 0, 256])
                hist2 = cv2.normalize(hist2, hist2).flatten()

                hist_comp = cv2.compareHist(hist1, hist2, method)
                histogram[m-1,n-1] = hist_comp

        
        img_out1 = histogram/np.linalg.norm(histogram)
        length1 = img_out1.shape[0]*img_out1.shape[1]
        vector_out1 = np.resize(img_out1,[1,length1])

        return(img_out1, vector_out1[0])

    def img_generator(self, positions):
        img_base = cv2.imread(self.img_base_path)
        conversion_factor = 330/1.227894 # radio del cirulo donde se posicionan los robots en px dividido por el radio en metros
        x_0 = 355 
        y_0 = 350 
        len_y, _, _ = img_base.shape

        for position in positions:
            x_meters,y_meters = position

            x_px = x_meters*conversion_factor
            y_px = y_meters*conversion_factor

            x = int(np.round(x_0 - x_px,0))
            y = int(np.round(y_0 + y_px,0))

            cv2.circle(img_base, (len_y-y,x), 10, (0,255,0), -1)

        return img_base

    def run_Argos(self, pfsm="--fsm-config --nstates 1 --s0 1", sleep=1):
        # Run argos to get features
        subprocess.run(f"cd {self.folder}/mission-folder; /home/jpmedina/argos3-installation/chocolate/ARGoS3-AutoMoDe/bin/automode_main -n -c {self.mission}.argos {pfsm}", shell=True)

    def computeMargin(self, init=False):
        """Implement 2) with SVM or projection algoritm
        """
        # svm algoritm
        svm = self.compute_SVM(self.muHistory, self.labelHistory)
        svmCoef = svm.coef_[0]
        SVs = svm.support_vectors_
        w = (np.array(svm.coef_[0])/LA.norm(svm.coef_[0], ord=2)).tolist()

        t = inf
        for i in range(1, len(self.muHistory)):
            mu = self.muHistory[i]
            tCand = np.dot(w, np.array(self.muE) - np.array(mu))
            if(tCand < t):
                t = tCand

        improved = False
        with open(f"{self.folder}/{self.experience}.json", "r") as file:
            info = json.load(file)
            previous_t = info[-1]["t"]
            previous_best = info[-1]["best fsm"]
            if(round(t,6) < previous_t):
                improved = True
        
        return w, t, svmCoef, SVs, improved, previous_best

    def compute_SVM(self, X, y):
        clf = svm.SVC(kernel="linear", class_weight='balanced')
        clf.fit(X, y)
        return clf

    def getFSM(self):
        with open(f"{self.folder}/mission-folder/fsm.txt",'r') as f:
            line = f.readline()
            PFSM = FSM.parse_from_commandline_args(line.split(" "))
        return PFSM

    def computePFSM(self):
        """Perform the algorithm to generate the desirable PFSM"""
        self.PFSM = self.getFSM()
        self.mu, phi_list = self.computeMu()
        self.muHistory.append(self.mu)
        self.labelHistory.append(-1)      
        w, t, svm, SVs, improved, previous_best = self.computeMargin()
        pfsm = " ".join(self.PFSM.convert_to_commandline_args())
        if(improved): 
            best_pfsm = " ".join(self.PFSM.convert_to_commandline_args())
        else:
            best_pfsm = previous_best

        print(f"iteration {self.iteration}")
        print(f"t = {t}")

        dico = {"iter" : self.iteration,
                "fsm" : pfsm,
                "best fsm" : best_pfsm,
                "mu" : [round(e,6) for e in self.mu],
                "phis" : [[round(e,6) for e in phi] for phi in phi_list],
                "w" : [round(e,6) for e in w],
                "svm coeff" : [round(e,6) for e in svm],
                "support vector(s)": SVs.tolist(), 
                "t" : round(t,6)}

        # Append JSON file history of produced PFSM
        with open(f"{self.folder}/{self.experience}.json", "r") as file:
            info = json.load(file)
        with open(f"{self.folder}/{self.experience}.json", "w") as file:
            info.append(dico)
            json.dump(info, file, indent=4)

        # Update mu history file
        with open(f"{self.folder}/muHistory",'a') as f:
            f.write(f"\n{[e for e in self.mu]};-1")

        # Update info for irace
        with open(f"{self.folder}/mission-folder/irl.txt",'w+') as f:
            f.write(f"{w}\n")
            f.write(f"{self.folder}/mission-folder/{self.mission}.argos\n")
            f.write(f"{self.img_base_path}")

    
if __name__ == '__main__':
    experience = sys.argv[1]
    iteration = sys.argv[2]
    autodemo = AUTODEMO(experience, iteration)
    PFSM = autodemo.computePFSM()
