import subprocess
from cmath import cos, exp, inf
import numpy as np
from numpy import linalg as LA
import random
import os
import ast
import sys
import json
import cv2
from datetime import datetime
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

    def __init__(self, experience):
        self.experience = experience
        self.mission = self.experience.split("_")[0]
        self.folder = f"/home/jpmedina/autodemo/irace/{experience}"
        self.demoFile = f"{self.folder}/mission-folder/{self.mission}.argos"
        self.arenaD = 3
        # self.demos = self.computeDemo(self.demoFile)
        # self.patches, self.obstacles = self.retrievePatches(self.demoFile)

        self.img_path_names = os.listdir(f"{self.folder}/mission-folder/demos/")
        self.img_base_path = f"{self.folder}/mission-folder/base/base_{self.mission}.png"
        self.img_base = cv2.imread(self.img_base_path)
        self.d_kernel = 100
        self.method = cv2.HISTCMP_BHATTACHARYYA

        self.muE = self.computeMuE()
        self.muHistory = []
        self.labelHistory = []
        self.muHistory.append(self.muE)
        self.labelHistory.append(1)

        self.PFSM = self.generate_rand_PFSM()
        self.BestPFSM = self.PFSM
        self.mu,_ = self.computeMu()
        self.muHistory.append(self.mu)
        self.labelHistory.append(-1)

        # Initialize mu history file
        with open(f"{self.folder}/muHistory",'w+') as f:
            f.write(f"{self.muE};1\n")
            f.write(f"{self.mu};-1")
  

    def computeMuE(self):
        phi_list = []
        
        for img_name in self.img_path_names:
            if img_name.endswith(".png"):
                img = cv2.imread(f"{self.folder}/mission-folder/demos/{img_name}")
                _, phi = self.histogram_comparision(img, self.img_base, d_kernel=self.d_kernel, stride=self.d_kernel+1, method=self.method)
                print(f"phiE: {phi}")
                phi_list.append(phi)

        mu = []
        for j in range(len(phi_list[0])):
            avg = 0
            for i in range(len(phi_list)):
                avg += phi_list[i][j]
            mu.append(avg/len(phi_list))
        
        print(f"muE = {mu}")

        return mu
    
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

        for position in positions:
            x_meters,y_meters = position

            x_px = x_meters*conversion_factor
            y_px = y_meters*conversion_factor

            x = int(np.round(x_0 + x_px,0))
            y = int(np.round(y_0 + y_px,0))

            cv2.circle(img_base, (x,y), 10, (0,255,0), -1)

        return img_base


    def run_Argos(self, pfsm="--fsm-config --nstates 1 --s0 1", sleep=1):
        # Run argos to get features
        subprocess.run(f"cd {self.folder}/mission-folder; /home/jpmedina/argos3-installation/chocolate/ARGoS3-AutoMoDe/bin/automode_main -n -c {self.mission}.argos {pfsm}", shell=True)       

    def generate_rand_PFSM(self):
        PFSM = FSM()
        PFSM.states.clear()
        n = random.randint(1,4)	# Number of states

        for i in range(n):
            behavior = Behavior.get_by_name(random.choice([b for b in Behavior.behavior_list]))  # b is just the name
            # set the behavior
            s = State(behavior)
            s.ext_id = i
            PFSM.states.append(s)

        for s in PFSM.states:
            m = random.randint(1,4)	# Number of states
            from_state = s
            for j in range(m):
                transition_id = s.ext_id
                transition_ext_id = transition_id	
                possible_states = [s for s in PFSM.states if s != from_state]
                if(possible_states):
                    to_state = random.choice(possible_states)
                    transition_condition = Condition.get_by_name(random.choice([c for c in Condition.condition_list]))
                    t = Transition(from_state, to_state, transition_condition)
                    t.ext_id = transition_ext_id
                    PFSM.transitions.append(t)

        PFSM.initial_state = [s for s in PFSM.states if s.ext_id == 0][0]
            
        return PFSM

    def computeMargin(self, init=False):
        """Implement 2) with SVM or projection algoritm
        """
        # svm algoritm
        svm = self.compute_SVM(self.muHistory, self.labelHistory)
        svmCoef = svm.coef_[0]
        SVs = svm.support_vectors_
        # TODO extract support vectors
        # check for fittimg warning

        w = (np.array(svm.coef_[0])/LA.norm(svm.coef_[0], ord=2)).tolist()

        t = inf
        for i in range(1, len(self.muHistory)):
            mu = self.muHistory[i]
            tCand = np.dot(w, np.array(self.muE) - np.array(mu))
            if(tCand < t):
                t = tCand

        return w, t, svmCoef, SVs

    def compute_SVM(self, X, y):
        clf = svm.SVC(kernel="linear", class_weight='balanced')
        clf.fit(X, y)
        return clf

    def initiateJson(self):
        """Perform the algorithm to generate the desirable PFSM"""
        pfsm = " ".join(self.PFSM.convert_to_commandline_args())
        bpfsm = " ".join(self.BestPFSM.convert_to_commandline_args())
        w, t, svm, SVs = self.computeMargin(init=True)
        print(f"iteration 0")
        print(f"w = {w}")
        print(f"t = {t}")

        now = datetime.now()
        dt = now.strftime("%d_%m_%Y_%H_%M_%S")

        with open(f"{self.folder}/{self.experience}.json", "w") as file:
            json.dump([], file)

        dico = {"iter" : "0",
                "begin time" : dt,
                "muE" : self.muE,
                "fsm" : pfsm,
                "best fsm": bpfsm,
                "mu" : [round(e,6) for e in self.mu],
                "w" : [round(e,6) for e in w],
                "svm coeff" : [round(e,6) for e in svm],
                "support vectors": SVs.tolist(),
                "t" : round(t,6),
                "Norm" : "L2"
                }

        with open(f"{self.folder}/{self.experience}.json", "r") as file:
            info = json.load(file)
        with open(f"{self.folder}/{self.experience}.json", "w") as file:
            info.append(dico)
            json.dump(info, file, indent=4)
        
        # Initial upload of irl information
        with open(f"{self.folder}/mission-folder/irl.txt",'w+') as f:
            f.write(f"{w}\n")
            f.write(f"{self.folder}/mission-folder/{self.mission}.argos")
        
if __name__ == '__main__':
    experience = sys.argv[1]
    autodemo = AUTODEMO(experience)
    autodemo.initiateJson()
