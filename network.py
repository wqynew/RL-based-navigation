import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np
import subprocess
import pdb
import torch.optim as optim
import torch.autograd as autograd
from multiprocessing_env import SubprocVecEnv
from scene_loader import ActiveVisionDatasetEnv
from absl import logging

import os

import pickle
from tensorboardX import SummaryWriter

CHECK_DIR='./checkpoint'


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
device = "cuda" if torch.cuda.is_available() else 'cpu'

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

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 5, 2, 1)))#3->2
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

        self.fpa=nn.Linear(7,512)
        self.fc50=nn.Linear(1536,1024)

        self.fc5=nn.Linear(1024,512)
        self.fc6=nn.Linear(512,256)

        self.actor=nn.Linear(256,7)
        self.critic=nn.Linear(256,1)

        self.fp1=nn.Linear(7,512)
        self.fp2=nn.Linear(1024,512)
        self.fp3=nn.Linear(512,512)
        self.fp_mean=nn.Linear(512,512)
        self.fp_sigma=nn.Linear(512,512)


    def forward_once(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l31(out)
        out = self.l32(out)
        out = self.l4(out)
        out=self.last(out)
        out = out.view(out.size(0), -1)
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
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))#512
        softplus = nn.Softplus()
        z_mean=self.fc_mean(x)
        z_sigma=softplus(self.fc_sigma(x))
        z=z_mean+torch.exp(z_sigma/2)*torch.randn(self.batch_size, 512, device=torch.device("cuda"))    
        z=F.relu(self.fz1(z))
        z=F.relu(self.fz2(z))
        pre_a=F.relu(self.fpa(pre_action)) 
        y=torch.cat((c_x,z,pre_a),1)
        y=F.relu(self.fc50(y))
        y=F.relu(self.fc5(y))
        y=F.relu(self.fc6(y))
        logit = self.actor(y)
        value = self.critic(y)
        return logit, value, z_mean,z_sigma, z 

    def chooseact(self, px1,px2,px3, x, g, deterministic=False):
        return True


    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)

    
    def act(self, px1,px2,px3, x, g,pre_action, deterministic=True):
        logit, value, z_mean, z_sigma, z  = self.forward(px1,px2,px3,x,g,pre_action)
        if not deterministic:
            _,action =torch.max(logit,1)
        else:
            probs = F.softmax(logit,dim=1)
            action = probs.multinomial(1).view(-1).data
        return action
    
    def evaluate_actions(self, px1, px2, px3, x, g,action,x_next,pre_action):
        logit, value, z_mean,z_sigma, z  = self.forward(px1,px2,px3,x,g,pre_action)
        probs     = F.softmax(logit,dim=1)
        log_probs = F.log_softmax(logit,dim=1)
        action_log_probs = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(1).mean()
        action = probs.multinomial(1).view(-1).data
        return logit, action_log_probs, value, entropy, z_mean,z_sigma, z,action 


class RolloutStorage(object):
    def __init__(self, num_steps, num_envs, state_shape):
        self.num_steps = num_steps
        self.num_envs  = num_envs
        self.pstates1  = torch.zeros(num_steps + 1, num_envs, *state_shape)
        self.pstates2  = torch.zeros(num_steps + 1, num_envs, *state_shape)
        self.pstates3  = torch.zeros(num_steps + 1, num_envs, *state_shape)
        self.pre_actions=torch.zeros(num_steps + 1, num_envs, 7)
        
        self.states  = torch.zeros(num_steps + 1, num_envs, *state_shape)
        self.gstates= torch.zeros(num_steps + 1, num_envs, *state_shape)

        self.rewards = torch.zeros(num_steps,     num_envs, 1)
        self.masks   = torch.ones(num_steps  + 1, num_envs, 1)
        self.actions = torch.zeros(num_steps,     num_envs, 1).long()
        self.gtactions = torch.zeros(num_steps,     num_envs, 1).long()
        self.gt_states=torch.zeros(num_steps,num_envs,*state_shape)
        self.use_cuda = False
            
    def cuda(self):
        self.use_cuda  = True
        self.pstates1    = self.pstates1.cuda()
        self.pstates2    = self.pstates2.cuda()
        self.pstates3    = self.pstates3.cuda()
        self.pre_actions=self.pre_actions.cuda()
        self.states    = self.states.cuda()
        self.gstates=self.gstates.cuda()

        self.rewards   = self.rewards.cuda()
        self.masks     = self.masks.cuda()
        self.actions   = self.actions.cuda()
        self.gtactions   = self.gtactions.cuda()
        self.gt_states=self.gt_states.cuda()

    def insert(self, step, pstate1,pstate2,pstate3,state, g,action, gtaction,gt_state, reward, mask,pre_action):
        self.pstates1[step + 1].copy_(pstate1)
        self.pstates2[step + 1].copy_(pstate2)
        self.pstates3[step + 1].copy_(pstate3)
        self.pre_actions[step+1].copy_(pre_action)

        self.states[step + 1].copy_(state)
        self.gstates[step + 1].copy_(g)
        self.actions[step].copy_(action)
        self.gtactions[step].copy_(gtaction)
        self.gt_states[step].copy_(gt_state)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)
        
    def after_update(self):
        self.pstates1[0].copy_(self.pstates1[-1])
        self.pstates2[0].copy_(self.pstates2[-1])
        self.pstates3[0].copy_(self.pstates3[-1])
        self.pre_actions[0].copy_(self.pre_actions[-1])
        self.states[0].copy_(self.states[-1])
        self.gstates[0].copy_(self.gstates[-1])
        self.masks[0].copy_(self.masks[-1])
        
    def compute_returns(self, next_value, gamma):
        returns   = torch.zeros(self.num_steps + 1, self.num_envs, 1)
        if self.use_cuda:
            returns = returns.cuda()
        returns[-1] = next_value
        for step in reversed(range(self.num_steps)):
            returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
        return returns[:-1]


def save_checkpoint(state, global_t):
    filename = 'checkpoint-{}.ckpt'.format(global_t)
    checkpoint_path = os.path.join(CHECK_DIR, filename)
    best_path = os.path.join(CHECK_DIR, 'best.ckpt')
    torch.save(state, best_path)
    torch.save(state, checkpoint_path)
    print('--- checkpoint saved to %s ---'.format(checkpoint_path))

def gau_kl(q_mu, q_sigma, p_mu, p_sigma):
    # https://github.com/openai/baselines/blob/f2729693253c0ef4d4086231d36e0a4307ec1cb3/baselines/acktr/utils.py
    num = (q_mu - p_mu)**2 + q_sigma**2 - p_sigma**2
    den = 2 * (p_sigma**2) + 1e-8
    kl = torch.mean(num/den + torch.log(p_sigma) - torch.log(q_sigma))
    return kl

def make_env():
    def _thunk():
        env = ActiveVisionDatasetEnv()
        return env
    return _thunk

if __name__ == "__main__":
    num_envs =6
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    state_shape = envs.observation_space.shape
    print("!!!state_shape:",state_shape)
    #a2c hyperparams:
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    num_steps = 10
    num_frames = int(10e7)

    #rmsprop hyperparams:
    lr    = 1e-4
    eps   = 1e-5
    alpha = 0.99

    #Init a2c and rmsprop

    actor_critic = Navigation()
    optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
        
    if USE_CUDA:
        actor_critic = actor_critic.cuda()

    rollout = RolloutStorage(num_steps, num_envs, envs.observation_space.shape)
    rollout.cuda()

    all_rewards = []
    all_losses  = []
    per_acc=[]
    shortest_path=[]
    episode_success=[]
    my_path=[]
    SPL=[]
    SR=[]
    pstate1,pstate2,pstate3,state,gstate,shortest,pre_action = envs.reset()
    print("shortest",shortest)
    for i in range(num_envs):
        shortest_path.append([])
        shortest_path[i].append(shortest[i])
        my_path.append([])
        my_path[i].append(1)
        episode_success.append([])
        episode_success[i].append(0)
    criterion = torch.nn.CrossEntropyLoss()
    criterionx = torch.nn.MSELoss()
    cos_loss=torch.nn.CosineSimilarity(dim=1,eps=1e-6)

    pstate1 = torch.FloatTensor(np.float32(pstate1))
    pstate2 = torch.FloatTensor(np.float32(pstate2))
    pstate3 = torch.FloatTensor(np.float32(pstate3))
    state = torch.FloatTensor(np.float32(state))
    gstate=torch.FloatTensor(np.float32(gstate))
    pre_action=torch.FloatTensor(np.float32(pre_action))

    rollout.pstates1[0].copy_(pstate1)
    rollout.pstates2[0].copy_(pstate2)
    rollout.pstates3[0].copy_(pstate3)
    rollout.states[0].copy_(state)
    rollout.gstates[0].copy_(gstate)
    rollout.pre_actions[0].copy_(pre_action)

    episode_rewards = torch.zeros(num_envs, 1)
    final_rewards   = torch.zeros(num_envs, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_path = os.path.join(CHECK_DIR, 'best.ckpt') 
    if os.path.isfile(best_path):      
        checkpoint = torch.load(best_path,map_location=device)
        global_t=checkpoint['global_t']
        actor_critic.load_state_dict(checkpoint['state_dict'],strict=False)

        print("=> loaded checkpoint '{}' (global_t {})"
            .format(best_path, checkpoint['global_t']))
    else:
        global_t=0
        print("=> no checkpoint found at '{}'".format(best_path))
    count=0
    running_corrects=0
    running_corrects1 =0
    writer=SummaryWriter()
    for i_update in range(num_frames):
        optimizer.zero_grad()
        for step in range(num_steps):
            action= actor_critic.act(Variable(pstate1),Variable(pstate2),Variable(pstate3),Variable(state),Variable(gstate),Variable(pre_action))
            #print("act:",action)
            pim1,pim2,pim3,next_state,g_state, reward, done,gt_action,gt_state, shortest,pre_action = envs.step(action.cpu().data.numpy())
            for i in range(num_envs):
                my_path[i][-1]+=1
                if reward[i]>5:
                    episode_success[i][-1]+=1
                if done[i]:
                    shortest_path[i].append(shortest[i])
                    episode_success[i].append(0)
                    my_path[i].append(1)
            #=====================================
            reward = torch.FloatTensor(reward).unsqueeze(1)
            episode_rewards += reward
            masks = torch.FloatTensor(1-np.array(done)).unsqueeze(1)
            count+=(1-masks).sum()
            #final_rewards *= masks
            final_rewards += (1-masks) * episode_rewards
            episode_rewards *= masks

            if USE_CUDA:
                masks = masks.cuda()
            pstate1 = torch.FloatTensor(np.float32(pim1))
            pstate2 = torch.FloatTensor(np.float32(pim2))
            pstate3 = torch.FloatTensor(np.float32(pim3))
            pre_action = torch.FloatTensor(np.float32(pre_action))
            state = torch.FloatTensor(np.float32(next_state))
            gstate = torch.FloatTensor(np.float32(g_state))
            gt_state=torch.FloatTensor(np.float32(gt_state))#ground_truth_state
            gt_action=torch.from_numpy(gt_action).to(device)

            rollout.insert(step,pstate1,pstate2,pstate3, state, gstate, action.unsqueeze(1).data, gt_action.unsqueeze(1).data,gt_state, reward, masks,pre_action)
        with torch.no_grad():
            _, next_value,_,_,_ = actor_critic(Variable(rollout.pstates1[-1]),Variable(rollout.pstates2[-1]),Variable(rollout.pstates3[-1]),Variable(rollout.states[-1]),Variable(rollout.gstates[-1]),Variable(rollout.pre_actions[-1]))#===============================
        with torch.set_grad_enabled(True):
            actor_critic.train()
            next_value = next_value.data
            returns = rollout.compute_returns(next_value, gamma)
            logit, action_log_probs, values, entropy, z_mean,z_sigma, z,myaction  = actor_critic.evaluate_actions(
                Variable(rollout.pstates1[:-1]).view(-1, *state_shape),
                Variable(rollout.pstates2[:-1]).view(-1, *state_shape),
                Variable(rollout.pstates3[:-1]).view(-1, *state_shape),
                Variable(rollout.states[:-1]).view(-1, *state_shape),
                Variable(rollout.gstates[:-1]).view(-1, *state_shape),
                Variable(rollout.actions).view(-1, 1),
                Variable(rollout.gt_states).view(-1, *state_shape),
                Variable(rollout.pre_actions[:-1]).view(-1, 7),
            )
            values = values.view(num_steps, num_envs, 1)
            advantages = Variable(returns) - values
            value_loss = advantages.pow(2).mean()
            action_loss1=criterion(logit,rollout.gtactions.view(-1))
            #======================================================
            gt_a=rollout.gtactions.view(-1).data.cpu()
            bt_size=gt_a.size()[0]
            gt_labels=[]
            for idx in range(bt_size):
                lvalu=np.array([0.,0.,0.,0.,0.,0.,0.],dtype=np.float32)
                lvalu[gt_a[idx]]=1.
                gt_labels.append(lvalu)
            gt_labels=torch.from_numpy(np.array(gt_labels)).cuda()
            #======================================================
            p_mean, p_sigma, pz=actor_critic.forward_prior(Variable(rollout.states[:-1]).view(-1, *state_shape),gt_labels)
            kl_loss = gau_kl(p_mean, p_sigma,z_mean, z_sigma) 
            recon_loss1= criterionx(z_mean, p_mean)
            recon_loss2= criterionx(z_sigma, p_sigma)
            xnext=actor_critic.forward_once(Variable(rollout.gt_states).view(-1, *state_shape))
            recon_loss=criterionx(xnext, z)+criterionx(xnext, pz)
            running_corrects += torch.sum(rollout.actions.view(-1).data.cpu() == gt_a)
            running_corrects1 += torch.sum(myaction.data.cpu() == gt_a)
            loss=action_loss1+(value_loss * value_loss_coef)+0.1*recon_loss+0.0001*kl_loss
            loss.backward()
            global_t+=1
            nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
            optimizer.step()

        if global_t%2==0:
            print("i_ipdates:",i_update,'| loss:', loss.item(), '| action_loss1:', action_loss1.item(),'| value_loss:', value_loss.item()) 
            print("kl_loss:",kl_loss.item(),'| recon_loss:',recon_loss.item(),'| recon_loss1:',recon_loss1.item(),'| recon_loss2:',recon_loss2.item())
            print("action:",rollout.actions.view(-1))
            print("paction:",myaction)
            print("gtaction:",rollout.gtactions.view(-1))
            acc=running_corrects.double()/((i_update+1)*num_steps*num_envs)
            acc1=running_corrects1.double()/((i_update+1)*num_steps*num_envs)
            print("accuracy:",acc)
            print("accuracy1:",acc1)
            writer.add_scalar('loss/all_loss',loss.item(),global_t)
            writer.add_scalar('loss/gt_action_loss',action_loss1.item(),global_t)
            writer.add_scalar('loss/kl-loss',kl_loss.item(),global_t)
            writer.add_scalar('loss/recon_loss',recon_loss.item(),global_t)
            writer.add_scalar('loss/value_loss',value_loss.item(),global_t)
            writer.add_scalar('loss/mean_loss',recon_loss1.item(),global_t)
            writer.add_scalar('loss/sigma_loss',recon_loss2.item(),global_t)
            writer.add_scalar('performance/per_acc',acc,global_t)
            writer.add_scalar('performance/per_acc1',acc1,global_t)
            
        if i_update%100==0:
            save_checkpoint({'global_t': global_t,'state_dict': actor_critic.state_dict()}, global_t)

        if count>0 and count% 100 == 0:
            v=final_rewards.sum()/count
            writer.add_scalar('performance/reward',v,global_t)
            for name, param in actor_critic.named_parameters():
                writer.add_histogram(name,param.clone().cpu().data.numpy(),global_t)
        rollout.after_update()  
        #============================================================
        if i_update>0 and i_update%1000==0:
            spl=[]
            sr=[]
            for i in range(num_envs):
                shp=np.array(shortest_path[i][:-1])
                myp=np.array(my_path[i][:-1])
                episode=np.array(episode_success[i][:-1])
                tv=(shp/myp)*episode
                elec=tv.shape[0]
                susc=np.count_nonzero(tv)
                if susc>0:
                    spl.append(tv.sum()/elec)#susc
                else:
                    spl.append(0)
                sr.append(susc/elec)   
            spl=np.array(spl)
            s=spl.sum()/num_envs
            SPL.append(s)
            sr=np.array(sr)
            v=sr.sum()/num_envs
            SR.append(v)
            writer.add_scalar('performance/SR',v,global_t)
            writer.add_scalar('performance/SPL',s,global_t)
            print("SPL:",SPL) 
            print("SR:",SR) 
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

