import subprocess
from cmath import cos, exp, inf
import numpy as np
from numpy import linalg as LA
import random
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

    def __init__(self, experience):
        self.experience = experience
        self.mission = self.experience.split("_")[0]
        self.folder = f"/home/jpmedina/autodemo/irace/{experience}"
        self.demoFile = f"{self.folder}/mission-folder/{self.mission}.argos"
        self.arenaD = 3
        self.demos = self.computeDemo(self.demoFile)
        self.patches, self.obstacles = self.retrievePatches(self.demoFile)

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

    def computeDemo(self, argosFile):
        # parse an xml file by name
        file = minidom.parse(f'{argosFile}')

        #use getElementsByTagName() to get tag
        models = file.getElementsByTagName('demo')

        positions = []
        for m in models:
            epucks = m.getElementsByTagName("epuck")
            pos = []
            for e in epucks:
                pos.append(ast.literal_eval("[" + e.getAttribute("position") + "]"))
            positions.append(pos)
        print(f"Demo pos: {positions}")
        
        return positions    

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

    def computeMuE(self):
        phi_list = []
        for demo in self.demos:
            phi = self.computePhiE(demo)
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

    def computePhiE(self, demo):
        phiE = []
        # distance to pathces features
        for p in self.patches:
            phi = []
            patch = p.copy()

            for pos in demo:
                if(len(patch) == 3):
                    distance = self.distToCircle(patch, pos)
                else:
                    distance = self.distToRect(patch, pos)
                phi.append(distance)

            h = (2*np.log(10))/(self.arenaD**2)
            phi = [exp(- h * self.arenaD * pos) for pos in phi]
            phi.sort(reverse=True) 

            for e in phi: phiE.append(e)

        
        phi = []
        for i in range(len(demo)):
            neighbors = demo.copy()
            neighbors.pop(i)
            distance = min([LA.norm(np.array(demo[i]) - np.array(n), ord=2) for n in neighbors])
            phi.append(distance)

        h = (2*np.log(10))/(self.arenaD**2)
        phi = [exp(- h * self.arenaD * pos) for pos in phi]
        phi.sort(reverse=True) 

        for e in phi: phiE.append(e)

        return phiE
    
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
