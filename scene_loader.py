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


ACTIONS=['RotateRight','RotateLeft','MoveAhead','MoveBack','MoveRight','MoveLeft','Stop']

wronglist=[2,5,15,8,202,203,207,308,311,415] #FloorPlan2,FloorPlan5,FloorPlan15
#FloorPlan202 FloorPlan203 FloorPlan207 FloorPlan308 FloorPlan311 FloorPlan415

class ActiveVisionDatasetEnv():
    """simulates the environment from ActiveVisionDataset."""
    cached_data=None
    def __init__(self,dataset_root='./aidata'):
        self.action_space =spaces.Discrete(7)#len(ACTIONS)
        self.observation_space=np.zeros([3,300,300])
        self._actions=ACTIONS   
        self.bc=ai2thor.controller.BFSController()
        self.bc.start()
        self.reset_num=0
        self.widx=10
        self.fidx=random.choice([0,2,3,4,2])

    def CheckAllGoal(self):
        wlist=[0,2,3,4]
        for i in wlist:
            for j in range(21,31):
                k=i*100+j
                k=5
                self._cur_world='FloorPlan'+str(k)
                print(self._cur_world)
                event=self.bc.reset(self._cur_world)
                self.bc.search_all_closed(self._cur_world)
                self.graph=self.bc.build_graph()
                n_points=len(self.bc.grid_points)
                for idx in range(n_points):
                    q=self.bc.key_for_point(self.bc.grid_points[idx])
                    if q not in self.graph:
                        print(self._cur_world,"false!!!")
                break
            break
        print("end!!")

    def rreset(self):
        if self.reset_num%100==0:
            self.fidx=random.choice([0,2,3,4])#[0,2,3,4]
            self.widx=random.randint(22,30)#0-19#(1,20)
            k=self.fidx*100+self.widx
            while k in wronglist:
                self.fidx=random.choice([0,2,3,4])#[0,2,3,4]
                self.widx=random.randint(22,30)#0-19#(1,20)
                k=self.fidx*100+self.widx
            self._cur_world='FloorPlan'+str(k)
            print(self._cur_world)
            event=self.bc.reset(self._cur_world)
            self.bc.search_all_closed(self._cur_world)
            self.graph=self.bc.build_graph()
            self.goalkeys=self.bc.gkeys
            self.tasktypenum=len(self.goalkeys)
            print(self.goalkeys)
            goalnum=0
            for ele in self.goalkeys:
                goalnum+=len(self.bc.goal[ele])
            if goalnum==0:
                self.reset()
        self.reset_num+=1
        self.frame=0
        self.reward=0
        self.isexist=0
        while self.isexist==0:
            gidx=random.randint(0,self.tasktypenum-1)
            self.goalname=self.goalkeys[gidx]
            self.isexist=len(self.bc.goal[self.goalname])
        n_points=len(self.bc.grid_points)
        cidx=random.randint(0,n_points-1)
        ri=random.randint(0,3)
        event = self.bc.step(dict(action='TeleportFull', x=self.bc.grid_points[cidx]['x'], y=self.bc.grid_points[cidx]['y'], z=self.bc.grid_points[cidx]['z'], rotation=90.0*ri, horizon=0.0))
        #**********
        event=self.bc.step(dict(action='RotateRight'))
        self.pim1=torch.Tensor(event.frame/255).transpose(0,2)
        event=self.bc.step(dict(action='RotateRight'))
        self.pim2=torch.Tensor(event.frame/255).transpose(0,2)
        event=self.bc.step(dict(action='RotateRight'))
        self.pim3=torch.Tensor(event.frame/255).transpose(0,2)
        event=self.bc.step(dict(action='RotateRight'))
        self.cimg=torch.Tensor(event.frame/255).transpose(0,2)
        #print(self.cimg)
        #print(self.cimg.size())
        #***********
        #print(self.isexist)
        #=========================================================================
        mindis=100
        for i in range(self.isexist):
            target=self.bc.goal[self.goalname][i]
            dis=target['distance']
            if dis<mindis:
                mindis=dis
                self.mygoal=target
                self.mygoal['cameraHorizon']=event.metadata['agent']['cameraHorizon']
        self.len_path=len(self.bc.shortest_plan(self.graph, event.metadata['agent'],self.mygoal))
        self.gimg=torch.Tensor(self.mygoal['frame']/255).transpose(0,2)
        #===============================================================================
        #self.len_path=100
        # for i in range(self.isexist):
        #     target=self.bc.goal[self.goalname][i]
        #     target['cameraHorizon']=event.metadata['agent']['cameraHorizon']
        #     path=self.bc.shortest_plan(self.graph, event.metadata['agent'],target)
        #     self.len_path=min(self.len_path,len(path))
        t=[0.,0.,0.,0.,0.,0.,0.]
        self.pre_action=torch.from_numpy(np.array(t,dtype=np.float32))
        return self.pim1,self.pim2,self.pim3, self.cimg, self.gimg, self.len_path,self.pre_action

    
    def reset(self):
        if self.reset_num%100==0:
            self.widx+=1
            if self.widx>21:
                self.widx=1
            k=self.fidx*100+self.widx
            while k in wronglist:
                self.widx+=1
                if self.widx>21:
                    self.widx=1
                k=self.fidx*100+self.widx
            self._cur_world='FloorPlan'+str(k)
            print(self._cur_world)
            event=self.bc.reset(self._cur_world)
            self.bc.search_all_closed(self._cur_world)
            self.graph=self.bc.build_graph()
            self.goalkeys=self.bc.gkeys
            self.tasktypenum=len(self.goalkeys)
            print(self.goalkeys)
            goalnum=0
            for ele in self.goalkeys:
                goalnum+=len(self.bc.goal[ele])
            if goalnum==0:
                self.reset()
        self.reset_num+=1
        self.frame=0
        self.reward=0
        self.isexist=0
        while self.isexist==0:
            gidx=random.randint(0,self.tasktypenum-1)
            self.goalname=self.goalkeys[gidx]
            self.isexist=len(self.bc.goal[self.goalname])
        n_points=len(self.bc.grid_points)
        cidx=random.randint(0,n_points-1)
        ri=random.randint(0,3)
        event = self.bc.step(dict(action='TeleportFull', x=self.bc.grid_points[cidx]['x'], y=self.bc.grid_points[cidx]['y'], z=self.bc.grid_points[cidx]['z'], rotation=90.0*ri, horizon=0.0))
        #**********
        event=self.bc.step(dict(action='RotateRight'))
        self.pim1=torch.Tensor(event.frame/255).transpose(0,2)
        event=self.bc.step(dict(action='RotateRight'))
        self.pim2=torch.Tensor(event.frame/255).transpose(0,2)
        event=self.bc.step(dict(action='RotateRight'))
        self.pim3=torch.Tensor(event.frame/255).transpose(0,2)
        event=self.bc.step(dict(action='RotateRight'))
        self.cimg=torch.Tensor(event.frame/255).transpose(0,2)
        #=================================================================================
        gidx=random.randint(0,self.isexist-1)
        self.mygoal=self.bc.goal[self.goalname][gidx]
        self.mygoal['cameraHorizon']=event.metadata['agent']['cameraHorizon']
        self.len_path=len(self.bc.shortest_plan(self.graph, event.metadata['agent'],self.mygoal))
        self.gimg=torch.Tensor(self.mygoal['frame']/255).transpose(0,2)
        t=[0.,0.,0.,0.,0.,0.,0.]
        self.pre_action=torch.from_numpy(np.array(t,dtype=np.float32))
        return self.pim1,self.pim2,self.pim3, self.cimg, self.gimg, self.len_path,self.pre_action

    def start(self):
        """Starts a new episode."""
        self.frame=0
        self.reward=0
        pim1,pim2,pim3, self.cimg, self.gimg,shortest,self.pre_action=self.reset()
        
        return pim1,pim2,pim3, self.cimg, self.gimg,shortest,self.pre_action
    
    def step(self, idx):#action is a digit
        t=[0.,0.,0.,0.,0.,0.,0.]
        t[idx]=1.
        self.pre_action=torch.from_numpy(np.array(t,dtype=np.float32))
        self.collided=False
        self.frame+=1
        self.done=False
        action=self._actions[idx]
        self.prev_event={'position':self.bc.last_event.metadata['agent']['position'],'rotation':self.bc.last_event.metadata['agent']['rotation'],'cameraHorizon':self.bc.last_event.metadata['agent']['cameraHorizon']}
        path=self.bc.shortest_plan(self.graph, self.prev_event,self.mygoal)
        self.len_path=len(path)
        if self.len_path>0:
            gt_action=path[0]['action']
            k=self._actions.index(gt_action)
            event=self.bc.step(dict(action=gt_action))
            self.gt_state=torch.Tensor(event.frame/255).transpose(0,2)
            gt_action=np.array(k)
            if k%2==0:
                actionx=self._actions[k+1]
                event=self.bc.step(dict(action=actionx))
            else:
                actionx=self._actions[k-1]
                event=self.bc.step(dict(action=actionx))
        else:
            gt_action=np.array(6)
            self.gt_state=self.cimg

        if action=='Stop':
            self.done=True
            event=self.bc.last_event
            issuccess=False
            for obj in event.metadata['objects']:#list
                if event.metadata['agent']['position']['x']==self.mygoal['position']['x'] and event.metadata['agent']['position']['z']==self.mygoal['position']['z'] and event.metadata['agent']['rotation']['y']==self.mygoal['rotation']['y']:
                #if obj['objectType']==self.goalname and obj['visible']:
                    self.reward=10.0
                    issuccess=True
                    break
            if not issuccess:
                self.reward=-5.0
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
                self.pim1=torch.Tensor(event.frame/255).transpose(0,2)
                event=self.bc.step(dict(action='RotateRight'))
                self.pim2=torch.Tensor(event.frame/255).transpose(0,2)
                event=self.bc.step(dict(action='RotateRight'))
                self.pim3=torch.Tensor(event.frame/255).transpose(0,2)
                event=self.bc.step(dict(action='RotateRight'))
                self.cimg=torch.Tensor(event.frame/255).transpose(0,2)
                q=self.bc.key_for_point(self.mygoal['position'])
                if q not in self.graph:
                    print(" q not in graph!!!")            
                path=self.bc.shortest_plan(self.graph, event.metadata['agent'],self.mygoal)
                len_p=len(path)
                if len_p<self.len_path:
                    self.reward=0.1-0.001
                else:
                    self.reward=-0.101
        if self.reward>0:
            self.gt_state=self.cimg
            gt_action=np.array(idx)
    
        if self.frame>100:
            self.done=True
        return self.pim1, self.pim2, self.pim3, self.cimg,self.gimg, self.reward,self.done, gt_action,self.gt_state, self.len_path,self.pre_action

    def observation(self):
        return (self.reward,self.done,
        self.cimg)
        

    

if __name__ == "__main__":
    env = ActiveVisionDatasetEnv()
    env.CheckAllGoal()
    print("end")



    







   



        












        




























