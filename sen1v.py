import collections
import copy
import json
import os
import time
import gym
from gym.envs.registration import register
import gym.spaces
from gym import spaces
import networkx as nx 
import numpy as np 
import scipy.io as sio 
from absl import logging
import sys
import cv2
import random
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
#import visualization_utils as vis_util 
import pickle
import ai2thor.controller
from numpy import linalg as LA
from PIL import Image
import torch 
import pickle


ACTIONS=['RotateRight','RotateLeft','MoveAhead','MoveBack','MoveRight','MoveLeft','Stop']
MOVE=['MoveAhead','MoveBack','MoveRight','MoveLeft']
wronglist=[2,5,15,8,202,203,207,308,311,415] #FloorPlan2,FloorPlan5,FloorPlan15
#FloorPlan202 FloorPlan203 FloorPlan207 FloorPlan308 FloorPlan311 FloorPlan415

class ActiveVisionDatasetEnv():
    """simulates the environment from ActiveVisionDataset."""
    cached_data=None
    def __init__(self,dataset_root='./aidata'):
        self.action_space =spaces.Discrete(7)#len(ACTIONS)
        self.observation_space=np.zeros([3,300,300])
        #self._dataset_root=dataset_root
        self._actions=ACTIONS   
        #print(self.tasktypenum)
        self.bc=ai2thor.controller.BFSController()
        self.bc.start()
        self.reset_num=0
        self._cur_world=None
        #===============================================================
        #self.reset()
    
    def reset(self,world,st_p,st_r,ed_p,ed_r,goalname):
        if self._cur_world!=world:
            self._cur_world=world
            print(self._cur_world)
            event=self.bc.reset(self._cur_world)
            self.bc.search_all_closed(self._cur_world)
            self.graph=self.bc.build_graph()
            self.goalkeys=self.bc.gkeys
            self.tasktypenum=len(self.goalkeys)
            print(self.goalkeys)
        self.frame=0
        self.reward=0
        self.goalname=goalname
        #=========================================================================
        #print("ed_r:",ed_r)
        event = self.bc.step(dict(action='TeleportFull', x=ed_p['x'], y=ed_p['y'], z=ed_p['z'], rotation=ed_r['y'], horizon=0.0))
        if not event.metadata['lastActionSuccess']:
            print("impossible 1")
        self.mygoal={'position':ed_p,'rotation':ed_r,'frame':event.frame,'cameraHorizon':event.metadata['agent']['cameraHorizon']}
        self.gimg=torch.Tensor(event.frame/255).transpose(0,2).cuda()
        #=========================================================================
        event = self.bc.step(dict(action='TeleportFull', x=st_p['x'], y=st_p['y'], z=st_p['z'], rotation=st_r, horizon=0.0))
        if not event.metadata['lastActionSuccess']:
            print("impossible 2")
        c_st=event.metadata['agent']['position']
        #**********
        event=self.bc.step(dict(action='RotateRight'))
        self.pim1=torch.Tensor(event.frame/255).transpose(0,2).cuda()
        event=self.bc.step(dict(action='RotateRight'))
        self.pim2=torch.Tensor(event.frame/255).transpose(0,2).cuda()
        event=self.bc.step(dict(action='RotateRight'))
        self.pim3=torch.Tensor(event.frame/255).transpose(0,2).cuda()
        event=self.bc.step(dict(action='RotateRight'))
        self.cimg=torch.Tensor(event.frame/255).transpose(0,2).cuda()
        st=event.metadata['agent']
        if st['position']==c_st:
            t=[0.,0.,0.,0.,0.,0.,0.]
            self.pre_action=torch.from_numpy(np.array(t,dtype=np.float32)).cuda()
            #===============================================================================
            return self.pim1,self.pim2,self.pim3, self.cimg, self.gimg, st,self.pre_action
        else:
            print("impossible 3")

    def start(self):
        """Starts a new episode."""
        self.frame=0
        self.reward=0
        pim1,pim2,pim3, self.cimg, self.gimg,shortest=self.reset()
        
        return pim1,pim2,pim3, self.cimg, self.gimg,shortest
    
    def step(self, idx):#action is a digit
        t=[0.,0.,0.,0.,0.,0.,0.]
        t[idx]=1.
        self.pre_action=torch.from_numpy(np.array(t,dtype=np.float32)).cuda()
        #print(idx)
        self.collided=False
        self.frame+=1
        self.done=False
        action=self._actions[idx]
        #lsts=self.bc.last_event.metadata['lastActionSuccess']
        self.prev_event={'position':self.bc.last_event.metadata['agent']['position'],'rotation':self.bc.last_event.metadata['agent']['rotation'],'cameraHorizon':self.bc.last_event.metadata['agent']['cameraHorizon']}

        #print(self._cur_world,"self.prev_event:",self.prev_event)
        if action=='Stop':
            self.done=True
            event=self.bc.last_event
            issuccess=False
            for obj in event.metadata['objects']:#list
                v=np.sqrt((self.bc.last_event.metadata['agent']['position']['x']-self.mygoal['position']['x'])**2+(self.bc.last_event.metadata['agent']['position']['z']-self.mygoal['position']['z'])**2)
                if obj['objectType']==self.goalname and obj['visible'] and v<0.51:
                    self.reward=10.0
                    issuccess=True
                    break
            if not issuccess:
                self.reward=-0.2
        else:
            event=self.bc.step(dict(action=action))# if collided, stay at the origin
            cur_event=event
            p=self.bc.key_for_point(event.metadata['agent']['position'])
            if p not in self.graph:
                print(" p not in graph!!!")
                self.reward=-0.2
                self.collided =True
                k=idx
                #back
                if k%2==0:
                    actionx=self._actions[k+1]
                    event=self.bc.step(dict(action=actionx))
                else:
                    actionx=self._actions[k-1]
                    event=self.bc.step(dict(action=actionx))
            elif not event.metadata['lastActionSuccess']:
                self.reward=-0.2
                self.collided =True
            else:
                event=self.bc.step(dict(action='RotateRight'))
                self.pim1=torch.Tensor(event.frame/255).transpose(0,2).cuda()
                event=self.bc.step(dict(action='RotateRight'))
                self.pim2=torch.Tensor(event.frame/255).transpose(0,2).cuda()
                event=self.bc.step(dict(action='RotateRight'))
                self.pim3=torch.Tensor(event.frame/255).transpose(0,2).cuda()
                event=self.bc.step(dict(action='RotateRight'))
                self.cimg=torch.Tensor(event.frame/255).transpose(0,2).cuda()
        if self.frame>100:
            self.done=True
        return self.pim1, self.pim2, self.pim3, self.cimg,self.gimg, self.reward,self.done,self.pre_action

    def observation(self):
        return (self.reward,self.done,
        self.cimg)
        

    

if __name__ == "__main__":
    env = ActiveVisionDatasetEnv()
    env.CheckAllGoal()
    print("end")



    







   



        












        




























