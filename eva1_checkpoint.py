from __future__ import print_function, division
"""
==========add LSTM vertion===================================
"""
from torch.autograd import Variable
from spectral import SpectralNorm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
import torch.nn.functional as F
import cv2
from pathlib import Path
import shutil

import multiprocessing as mp
import threading as td

import numpy as np


import torch.autograd as autograd

from multiprocessing_env import SubprocVecEnv
from sen1v import ActiveVisionDatasetEnv
from torch.autograd import Variable as V
import futil
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Navigation(nn.Module):
    def __init__(self, batch_size=64, image_size=300, conv_dim=32):
        super(Navigation, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        layer31 = []
        layer32 = []
 
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 5, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 5, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 5, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer31.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 5, 2, 1)))
        layer31.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer32.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 5, 2, 1)))
        layer32.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
 
        if self.imsize == 300:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 5, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l31 = nn.Sequential(*layer31)
        self.l32 = nn.Sequential(*layer32)


        last.append(nn.Conv2d(curr_dim, curr_dim, 3))
        self.last = nn.Sequential(*last)#512
        #===========================================================================================
        self.f0=nn.Linear(1024,1024)
        self.f1=nn.Linear(1024,512)
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,512)

        self.fc3=nn.Linear(512*4,1024)
        self.fc4=nn.Linear(1024,512)
        self.fc_mean=nn.Linear(512,512)
        self.fc_sigma=nn.Linear(512,512)
        self.fz1=nn.Linear(512,512)
        self.fz2=nn.Linear(512,512)
        #*******************
        self.fpa=nn.Linear(7,512)
        self.fc50=nn.Linear(1536,1024)
        #********************
        self.fc5=nn.Linear(1024,512)
        self.fc6=nn.Linear(512,256)
        #=============================================================================================
        self.actor=nn.Linear(256,7)
        self.critic=nn.Linear(256,1)
        #==============================================================================================
        self.fp1=nn.Linear(7,512)
        self.fp2=nn.Linear(1024,512)
        self.fp3=nn.Linear(512,512)
        self.fp_mean=nn.Linear(512,512)
        self.fp_sigma=nn.Linear(512,512)


    def forward_once(self, x):
        #print("forward_once:",x.size())
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        #print("l3:",out.size())
        out = self.l31(out)
        #print("l31:",out.size())
        out = self.l32(out)
        #print("l32:",out.size())
        out = self.l4(out)
        #print("l4:",out.size())
        out=self.last(out)
        #print("last:",out.size())
        out = out.view(out.size(0), -1)
        #print(out.size())
        out=F.relu(self.f0(out))
        out=F.relu(self.f1(out))
        return out

    def forward_prior(self,x,a):
        x=self.forward_once(x)
        
        softplus = nn.Softplus()
        pa=F.relu(self.fp1(a))
        px=torch.cat((x,pa),1)
        px=F.relu(self.fp2(px))
        px=F.relu(self.fp3(px))
        p_mean=self.fp_mean(px)
        p_sigma=softplus(self.fp_sigma(px))
        self.batch_size=p_mean.size()[0]
        pz=p_mean+torch.exp(p_sigma/2)*torch.randn(self.batch_size, 512, device=torch.device("cuda")) 
        pz=F.relu(self.fz1(pz))
        pz=F.relu(self.fz2(pz))
        return p_mean,p_sigma, pz

    def forward(self, px1,px2,px3, x,g,pre_action):
        self.batch_size=x.size()[0]
        px1 = self.forward_once(px1)
        px2 = self.forward_once(px2)
        px3 = self.forward_once(px3)
        c_x = self.forward_once(x)
        x=c_x
        g =self.forward_once(g)
#=================================
        px1=torch.cat((px1,g),1)
        px1=F.relu(self.fc1(px1))
        px1=F.relu(self.fc2(px1))
        px2=torch.cat((px2,g),1)
        px2=F.relu(self.fc1(px2))
        px2=F.relu(self.fc2(px2))
        px3=torch.cat((px3,g),1)
        px3=F.relu(self.fc1(px3))
        px3=F.relu(self.fc2(px3))
        x=torch.cat((x,g),1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=torch.cat((px1,px2,px3,x),1)
#=======================================================================
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))#512
        softplus = nn.Softplus()
        z_mean=self.fc_mean(x)
        z_sigma=softplus(self.fc_sigma(x))
        z=z_mean+torch.exp(z_sigma/2)*torch.randn(self.batch_size, 512, device=torch.device("cuda"))    
        z=F.relu(self.fz1(z))
        z=F.relu(self.fz2(z))
#=======================================================================
        pre_a=F.relu(self.fpa(pre_action)) 
        y=torch.cat((c_x,z,pre_a),1)
        y=F.relu(self.fc50(y))
        y=F.relu(self.fc5(y))
        y=F.relu(self.fc6(y))
#======================================
        logit = self.actor(y)
        value = self.critic(y)
        return logit, value, z_mean,z_sigma, z 

    def chooseact(self, px1,px2,px3, x, g, deterministic=False):
        return True


    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)

    
    def act(self, px1,px2,px3, x, g,pre_action, deterministic=True):
        logit, value, z_mean, z_sigma, z  = self.forward(px1,px2,px3,x,g,pre_action)

        probs = F.softmax(logit,dim=1)
        action = probs.multinomial(1).view(-1).data
        return action
    
    def evaluate_actions(self, px1, px2, px3, x, g,action,x_next,pre_action):
        logit, value, z_mean,z_sigma, z  = self.forward(px1,px2,px3,x,g,pre_action)
        probs     = F.softmax(logit,dim=1)
        #print("probs:",probs)
        log_probs = F.log_softmax(logit,dim=1)
        action_log_probs = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(1).mean()
  
        #===========================================================================
        #_,action =torch.max(logit,1)
        action = probs.multinomial(1).view(-1).data
        return logit, action_log_probs, value, entropy, z_mean,z_sigma, z,action 
      
#=====================================================================================================
num_actions = 7

actor_critic =Navigation()
USE_CUDA=True
if USE_CUDA:
    actor_critic  = actor_critic.cuda()

CHECK_DIR="./checkpoint/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from os import listdir
checfile=listdir('./checkpoint')

print(checfile)
MOVE=['MoveAhead','MoveBack','MoveRight','MoveLeft']
SR=[]
SPL=[]

tenv = ActiveVisionDatasetEnv()
testpath=[]
with open('./test/evaluation2/testpath1','rb') as fp:
    testpath1=pickle.load(fp)
testpath=testpath1
with open('./test/evaluation2/testpath2','rb') as fp:
    testpath2=pickle.load(fp)
testpath.extend(testpath2)
with open('./test/evaluation2/testpath3','rb') as fp:
    testpath3=pickle.load(fp)
testpath.extend(testpath3)
with open('./test/evaluation2/testpath4','rb') as fp:
    testpath4=pickle.load(fp)
testpath.extend(testpath4)
print("test path:",len(testpath))
max_len=0
for fele in checfile:
    fele='best.ckpt'
    best_path = os.path.join(CHECK_DIR, fele) #checkpoint-1186703
    if os.path.isfile(best_path):      
        checkpoint = torch.load(best_path,map_location=device)
        global_t=checkpoint['global_t']
        actor_critic.load_state_dict(checkpoint['state_dict'],strict=False)#====================================
        print("=> loaded checkpoint '{}' (global_t {})"
            .format(best_path, checkpoint['global_t']))
    else:
        global_t=0
        print("=> no checkpoint found at '{}'".format(best_path))
    #===================================================
    successtime=0
    spl=0
    l5_p=0
    l5_s=0
    l5_spl=0
    mid_p=[0,0,0,0,0,0,0,0,0,0,0,0]
    mid_s=[0,0,0,0,0,0,0,0,0,0,0,0]
    mid_spl=[0,0,0,0,0,0,0,0,0,0,0,0]
    success_dis_to_goal=[]
    all_dis_to_goal=[]
    ori_dis_to_goal=[]
    task_actions={}
    collide_actions={}
    room_idx=[22,24,26,28,30]
    with torch.set_grad_enabled(False):
    #=====================================================================================================================================
        count=0
        for tp in testpath:
            world=tp['world']
            st_p=tp['st_position']
            ed_p=tp['ed_position']
            st_r=tp['st_rotation']
            ed_r=tp['ed_rotation']
            goalname=tp['goalname']
            current_pim1,current_pim2,current_pim3,current_state,current_g,start_agent,pre_action = tenv.reset(world,st_p,st_r,ed_p,ed_r,goalname)
            step=0
            slen=len(tenv.bc.shortest_plan(tenv.graph, start_agent,tenv.mygoal))
            ori_dis_to_goal.append(slen)
            if slen>max_len:
                max_len=slen
# We check the code in [1] and find their shortest path represents the nodes that need to be visited (e.g., 1->2->3),including the start and the end. 
# Our shortest path represents the action sequence without stop (e.g., ->,->), and hence we set the threshold $4$ below in accordance with the $5$ in [1].
#[1]Learning to Learn How to Learn: Self-Adaptive Visual Navigation using Meta-Learning
            if slen>=4:
                l5_s+=1
            idx=(slen+1)//5
            mid_s[idx]+=1
            path=None
            task_actions[str(count)]=[]
            collide_actions[str(count)]=[]
            colli_count=0
            while step<100:
                if step%5==0 and step<40:
                    task_actions[str(count)].append(step)
                    collide_actions[str(count)].append(colli_count)
                action = actor_critic.act(current_pim1.unsqueeze(0),current_pim2.unsqueeze(0),current_pim3.unsqueeze(0),current_state.unsqueeze(0),current_g.unsqueeze(0),pre_action.unsqueeze(0))
                current_pim1,current_pim2,current_pim3,current_state,current_g, reward, done,pre_action= tenv.step(action.cpu().data.numpy()[0])
                step+=1 
                if tenv.collided:
                    colli_count+=1
                if done and tenv.reward>5.0:
                    mylen=step
                    v=slen/max(mylen,slen)
                    spl+=v
                    if slen>=4:
                        l5_p+=1
                        l5_spl+=v 
                    mid_p[idx]+=1
                    mid_spl[idx]+=v                  
                    #================================================
                    path=tenv.bc.shortest_plan(tenv.graph, tenv.bc.last_event.metadata['agent'],tenv.mygoal)
                    # pcount=0
                    # for pele in path:
                    #     act=pele['action']
                    #     if act in MOVE:
                    #         pcount+=1
                    success_dis_to_goal.append(len(path))
                    successtime+=1
                    break
                elif done:
                    break
            if step<40:
                task_actions[str(count)].append(step)
                collide_actions[str(count)].append(colli_count)
            count+=1
            if not path:
                path=tenv.bc.shortest_plan(tenv.graph, tenv.bc.last_event.metadata['agent'],tenv.mygoal)
                # pcount=0
                # for pele in path:
                #     act=pele['action']
                #     if act in MOVE:
                #         pcount+=1
            all_dis_to_goal.append(len(path))
        spl=spl/len(testpath)
        successtime=successtime/len(testpath)
        if l5_s>0:
            l5_sr=l5_p/l5_s
            l5_spl=l5_spl/l5_s
            print("l5_sr:",l5_sr)
            print("l5_spl:",l5_spl)
        print("SPL:",spl)
        print("SR:",successtime) 
        print("success_dis_to_goal:",success_dis_to_goal)
        print("all_dis_to_goal:",all_dis_to_goal)
        print("ori_dis_to_goal:",ori_dis_to_goal)
        ratio={}
        for i in range(8):
            s_act=0
            c_act=0
            for j in range(len(testpath)):
                l=len(task_actions[str(j)])
                if i<l:
                    s_act+=task_actions[str(j)][i]
                    c_act+=collide_actions[str(j)][i]
            if s_act>0:
                ratio[str(i)]=c_act/s_act
            else:
                ratio[str(i)]=0
        print("Failed ratio:",ratio)
        for i in range(9):
            if mid_s[i]>0:
                sr=mid_p[i]/mid_s[i]
                spl=mid_spl[i]/mid_s[i]
                print("int:",i,"| SR:",sr,"|SPL",spl)                
        print("max_len:",max_len)



    




    
    





