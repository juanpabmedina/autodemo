import numpy as np
import sys
import pandas as pd
import ast
import subprocess
import os
import stat
import ast
import time
import json
import cv2
import matplotlib.pyplot as plt
from xml.dom import minidom
from math import exp, sqrt, sin, cos
from cmath import inf
from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment


def run_Argos(fsm, argos):
    # run argos with passed fsm to write hitory file
    with open("./argos.sh",'w+') as f:
        f.write("#!/usr/bin/env python\n")
        f.write(f"/home/robotmaster/ARGoS3-AutoMoDe/bin/automode_main -n -c {argos} --fsm-config {fsm}\n")

    st = os.stat('./argos.sh')
    os.chmod('./argos.sh', st.st_mode | stat.S_IEXEC)

    subprocess.run(["bash", 'argos.sh'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def img_generator(positions, img_base_path):
    img_base = cv2.imread(img_base_path)
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

exp_name = 'aac_26_09'
os.makedirs(f'/home/robotmaster/autodemo_local/imgResults/{exp_name}',exist_ok=True)

paths = os.listdir(f'/home/robotmaster/autodemo_local/JSON/{exp_name}')
t_idx_tot = []
for path in paths: 
    f = open(f'/home/robotmaster/autodemo_local/JSON/{exp_name}/{path}')
    data = json.load(f)

    t_tot = []
    for i in range(len(data)):
        t = data[i]["t"]
        t_tot.append(t)
    t = min(t_tot)
    t_idx = np.argmin(t_tot)
    t_idx_tot.append(t_idx)

    name = exp_name.split("_")[0]
    fsm = data[-1]['best fsm']
    # fsm = data[-1]['best fsm']

    # argos = f"/home/robotmaster/autodemo_local/argosFiles/80Robots/{name}.argos"
    argos = f"/home/robotmaster/autodemo_local/argosResults/{name}.argos"
    inicio = time.time()
    run_Argos(fsm, argos)
    fin = time.time()

    print(fin-inicio) 

    swarm_pos = []
    with open(f"/home/robotmaster/autodemo_local/pos.mu", "r") as f:
        for pos in f.readlines():
            swarm_pos.append(ast.literal_eval("[" + pos[:-2] + "]"))
        #print(f"swarm pos: {swarm_pos}")

    img_base_path = f"/home/robotmaster/autodemo_local/baseImg/base_{name}.png"
    img = img_generator(swarm_pos, img_base_path)
    img_name = path.split(".")[0]
    cv2.imwrite(f"/home/robotmaster/autodemo_local/imgResults/{exp_name}/{img_name}.png", img)

print(np.mean(t_idx_tot))

score = open(f'/home/robotmaster/autodemo_local/score.txt')
data_score = []

with open(f'/home/robotmaster/autodemo_local/score.txt', 'r') as f:
    for score in f.readlines():
        data_score.append(ast.literal_eval(score[:-1]))

# dict_score = {'name': data_score}

# df_score = pd.DataFrame.from_dict(dict_score)
# df_score.to_csv('../score.csv',index=False)


print(data_score)
csv_file = './score.csv'
df = pd.read_csv(csv_file)
df[f'{exp_name}'] = pd.Series(data_score)
df.to_csv(csv_file, index=False)
os.remove('./score.txt')
