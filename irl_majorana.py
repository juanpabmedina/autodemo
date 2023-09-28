import numpy as np
import sys
import logging
import ast
import subprocess
import os
import stat
import ast
import time
from xml.dom import minidom
from math import exp, sqrt, sin, cos
from cmath import inf
from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment

from automode.architecture.finite_state_machine import State, Transition, FSM
from automode.modules.chocolate import Behavior, Condition

def distToCircle(circle, pos, obstacles):
    c_x = circle[0]
    c_y = circle[1]
    r = circle[2]
    for obs in obstacles:
        if(intersect(pos,circle,obs[0], obs[1])):
            return 3
    return max(0, sqrt((pos[0]-c_x)**2 + (pos[1] - c_y)**2) - r)

def distToRect(rect, pos, obstacles):
    x_min = rect[0] - rect[2]/2
    x_max = rect[0] + rect[2]/2
    y_min = rect[1] - rect[3]/2
    y_max = rect[1] + rect[3]/2

    dx = max(x_min - pos[0], 0, pos[0] - x_max)
    dy = max(y_min - pos[1], 0, pos[1] - y_max)
    for obs in obstacles:
        if(intersect(pos,[x_min,pos[1]],obs[0], obs[1]) or
           intersect(pos,[x_max,pos[1]],obs[0], obs[1]) or
           intersect(pos,[pos[0],y_min],obs[0], obs[1]) or
           intersect(pos,[pos[0],y_max],obs[0], obs[1])):
            return 3
    return sqrt(dx**2 + dy**2)

def ccw(a, b, c):
    return (c[0] - a[0])*(b[1] - a[1]) > (b[0] - a[0])*(c[1] - a[1])

# Return true if segments AB and CD intersect
def intersect(a, b, c, d):
    return (ccw(a,c,d) != ccw(b,c,d)) and (ccw(a,b,c) != ccw(a,b,d))

def compute_phi(fsm, argos):
    """ Compute a list of info relative to the fsm
    firsts elements are states and conditions of the fsm
    before last element is the fsm in command line for argos
    last element is swarm sms dictionary 
    """
    # parse an xml file by name
    file = minidom.parse(f'{argos}')

    #use getElementsByTagName() to get tag
    patches = []

    #retriving circle patches
    circles = file.getElementsByTagName('circle')
    whiteCircles = []
    blackCircles = []
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

    # retrive all obstacles
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

    succeed = False
    while(not succeed):
        swarm_pos = []
        run_Argos(fsm, argos)
        with open("./pos.mu", "r") as f:
            for pos in f.readlines():
                try:
                    line = ast.literal_eval("[" + pos[:-2] + "]")
                    if(len(line) == 2):
                        swarm_pos.append(line)
                except:
                    print("misshape line")
            if(len(swarm_pos) == 40):
                succeed = True
         
    phiTot = []
    for p in patches:
        phi = []
        patch = p.copy()
        
        for pos in swarm_pos:
            min_distance = inf
            if(len(patch) == 3):
                distance = distToCircle(patch, pos, obstacles)
            else:
                distance = distToRect(patch, pos, obstacles)
            phi.append(distance)
    
        h = (2*np.log(10))/(3**2)
        phi = [exp(- h * 3 * pos) for pos in phi]
        phi.sort(reverse=True)

        for e in phi: phiTot.append(e)

    
    phi = []
    for i in range(len(swarm_pos)):
        neighbors = swarm_pos.copy()
        neighbors.pop(i)
        distance = min([LA.norm(np.array(swarm_pos[i]) - np.array(n), ord=2) for n in neighbors])
        phi.append(distance)

    h = (2*np.log(10))/(3**2)
    phi = [exp(- h * 3 * pos) for pos in phi]
    phi.sort(reverse=True)

    for e in phi: phiTot.append(e)
    
    return phiTot

def run_Argos(fsm, argos):
    # run argos with passed fsm to write hitory file
    with open("./argos.sh",'w+') as f:
        f.write("#!/usr/bin/env python\n")
        f.write(f"/home/jpmedina/argos3-installation/chocolate/ARGoS3-AutoMoDe/bin/automode_main -n -c {argos} --fsm-config {fsm}\n")

    st = os.stat('./argos.sh')
    os.chmod('./argos.sh', st.st_mode | stat.S_IEXEC)

    subprocess.run(["bash", 'argos.sh'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def computeScore(fsm, w, argos):
    phi = compute_phi(fsm, argos)
    score = -np.dot(np.array(w).T, phi)
    return score

if __name__ == '__main__':
    with open("../mission-folder/irl.txt",'r') as f:
        w = ast.literal_eval(f.readline())
        argos = f.readline()

    fsm = sys.argv[1]
    score = computeScore(fsm, w, argos)

    print(f"Score {score}")
