import subprocess
from cmath import cos, exp, inf
import numpy as np
from numpy import linalg as LA
import os
import ast
import sys
import json
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

    def __init__(self, experience, iteration):
        self.iteration = iteration
        self.experience = experience
        self.mission = self.experience.split("_")[0]
        self.folder = f"/home/jpmedina/originalAutodemo/autodemo/irace/{experience}"
        self.demoFile = f"{self.folder}/mission-folder/{self.mission}.argos"
        self.arenaD = 3
        self.patches, self.obstacles = self.retrievePatches(self.demoFile)

        # parse mu history for svm from muHistory.txt
        self.muHistory = []
        self.labelHistory = []
        with open(f"{self.folder}/muHistory", "r") as f:
            lines = f.readlines()
            for line in lines:
                self.muHistory.append(ast.literal_eval(line.split(";")[0]))
                self.labelHistory.append((int(line.split(";")[1])))
        self.muE = self.muHistory[0]

    def retrievePatches(self, argosFile):
        patches = []

        # parse an xml file by name
        file = minidom.parse(f'{argosFile}')

        #retriving circle patches
        circles = file.getElementsByTagName('circle')
        for c in circles:
            if(c.getAttribute("color") == "white"):
                patches.append(ast.literal_eval("[" + c.getAttribute("position") + "," + c.getAttribute("radius") + "]"))
            else:
                patches.append(ast.literal_eval("[" + c.getAttribute("position") + "," + c.getAttribute("radius") + "]"))


        #retriving rect patches
        rectangles = file.getElementsByTagName('rectangle')
        whiteRectangles = []
        blackRectangles = []
        for r in rectangles:
            if(r.getAttribute("color") == "white"):
                patches.append(ast.literal_eval("[" + r.getAttribute("center") + "," + r.getAttribute("width") + "," + r.getAttribute("height") + "]"))
            else:
                patches.append(ast.literal_eval("[" + r.getAttribute("center") + "," + r.getAttribute("width") + "," + r.getAttribute("height") + "]"))

        obstacles = []
        boxes = file.getElementsByTagName('box')
        for b in  boxes:
            if("obstacle" in b.getAttribute("id")):
                body = b.getElementsByTagName("body")[0]
                center = ast.literal_eval("[" + body.getAttribute("position") + "]")[:-1]
                width = ast.literal_eval("[" + b.getAttribute("size") + "]")[1]
                orientation = ast.literal_eval("[" + body.getAttribute("orientation") + "]")[0]
                a = [center[0] + width*sin(orientation), center[1] + width*cos(orientation)]
                b = [center[0] - width*sin(orientation), center[1] - width*cos(orientation)]
                obstacles.append([a,b])

        return patches, obstacles

    def distToCircle(self, circle, pos):
        c_x = circle[0]
        c_y = circle[1]
        r = circle[2]
        for obs in self.obstacles:
            if(self.intersect(pos,circle,obs[0], obs[1])):
                return self.arenaD
        return max(0, sqrt((pos[0]-c_x)**2 + (pos[1] - c_y)**2) - r)

    def distToRect(self, rect, pos):
        x_min = rect[0] - rect[2]/2
        x_max = rect[0] + rect[2]/2
        y_min = rect[1] - rect[3]/2
        y_max = rect[1] + rect[3]/2

        dx = max(x_min - pos[0], 0, pos[0] - x_max)
        dy = max(y_min - pos[1], 0, pos[1] - y_max)
        
        for obs in self.obstacles:
            if(self.intersect(pos,[x_min,pos[1]],obs[0], obs[1]) or
               self.intersect(pos,[x_max,pos[1]],obs[0], obs[1]) or
               self.intersect(pos,[pos[0],y_min],obs[0], obs[1]) or
               self.intersect(pos,[pos[0],y_max],obs[0], obs[1])):
               return self.arenaD
        return sqrt(dx**2 + dy**2)

    def ccw(self, a, b, c):
        return (c[0] - a[0])*(b[1] - a[1]) > (b[0] - a[0])*(c[1] - a[1])

    # Return true if segments AB and CD intersect
    def intersect(self, a, b, c, d):
        return (self.ccw(a,c,d) != self.ccw(b,c,d)) and (self.ccw(a,b,c) != self.ccw(a,b,d))
    
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

        phiTot = []
        for p in self.patches:
            phi = []
            patch = p.copy()

            for pos in swarm_pos:
                if(len(patch) == 3):
                    distance = self.distToCircle(patch, pos)
                else:
                    distance = self.distToRect(patch, pos)
                phi.append(distance)

            h = (2*np.log(10))/(self.arenaD**2)
            phi = [exp(- h * self.arenaD * pos) for pos in phi]
            phi.sort(reverse=True) 

            for e in phi: phiTot.append(e)

        
        phi = []
        for i in range(len(swarm_pos)):
            neighbors = swarm_pos.copy()
            neighbors.pop(i)
            distance = min([LA.norm(np.array(swarm_pos[i]) - np.array(n), ord=2) for n in neighbors])
            phi.append(distance)

        h = (2*np.log(10))/(self.arenaD**2)
        phi = [exp(- h * self.arenaD * pos) for pos in phi]
        phi.sort(reverse=True) 

        for e in phi: phiTot.append(e)
        
        
        return phiTot

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

        now = datetime.now()
        dt = now.strftime("%d_%m_%Y_%H_%M_%S")

        dico = {"iter" : self.iteration,
                "fsm" : pfsm,
                "time" : dt,
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
            f.write(f"{self.folder}/mission-folder/{self.mission}.argos")
    
if __name__ == '__main__':
    experience = sys.argv[1]
    iteration = sys.argv[2]
    autodemo = AUTODEMO(experience, iteration)
    PFSM = autodemo.computePFSM()
