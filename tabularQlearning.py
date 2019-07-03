import os
import numpy as np
import params_priors as p
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
#import priors_tabular as PR

def Qlearn_multirun_tab():
	#This function just runs multiple instances of 
	#Q-learning. Doing so helps obtain an average performance 
	#measure over multiple runs.
	retlog=[] # log of returns of all episodes, in all runs
	for i in range(p.Nruns):
		print("Run no:",i)
		Q,ret=main_Qlearning_tab()#call Q learning
		if i==0:
			retlog=ret
		else:
			retlog=np.vstack((retlog,ret))
		#retlog.append(ret)
		if (i+1)/p.Nruns==0.25:
			print('25% runs complete')
		elif (i+1)/p.Nruns==0.5:
			print('50% runs complete')
		elif (i+1)/p.Nruns==0.75:
			print('75% runs complete')
		elif (i+1)==p.Nruns:
			print('100% runs complete')
	meanreturns=(np.mean(retlog,axis=0))
	plt.plot(meanreturns)
	plt.show()
	return Q, retlog

def main_Qlearning_tab():
	#This calls the main Q learning algorithm
	Q=np.zeros((p.a,p.b,p.A)) # initialize Q function as zeros
	goal_state=p.targ#target point
	returns=[]#stores returns for each episode
	ret=0
	for i in range(p.episodes):
		if (i+1)/p.episodes==0.25:
			print('25% episodes done')
		elif (i+1)/p.episodes==0.5:
			print('50% episodes done')
		elif (i+1)/p.episodes==0.75:
			print('75% episodes done')
		elif (i+1)/p.episodes==1:
			print('100% episodes done')
		if i%1==0:
			returns.append(calcret(Q,goal_state,i))#compute return offline- can also be done online, but this way, a better estimate can be obtained
		Q=Qtabular(Q,i)#call Q learning
	return Q, returns

def Qtabular(Q,episode_no):
	#This is the main Q learning algorithm
	initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])# set initial state randomly
	rounded_initial_state=staterounding(initial_state)
	target_state=p.targ# set target state
	while p.world[rounded_initial_state[0],rounded_initial_state[1]]==1 or np.linalg.norm(rounded_initial_state-target_state)<=p.thresh:# make sure initial state is not an obstacle
		initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
		rounded_initial_state=staterounding(initial_state)
	state=initial_state.copy()
	roundedstate=staterounding(state)#because we only consider discretized states here
	count=0
	breakflag=0
	ret=0
	eps_live=p.epsilon#1-(p.epsilon_decay*episode_no)#exploration parameter. Decreases with time.
	#another option is to set eps_live=1-(p.epsilon_decay*episode_no)#decaying epsilon
	while np.linalg.norm(roundedstate-target_state)>p.thresh:#check if goal is reached
		if breakflag==1:
			break

		if eps_live>np.random.sample():#epsilon greedy exploration
			#explore
			a=np.random.randint(p.A)#random action
		else:
			#exploit
			Qmax,a=maxQ_tab(Q,state)#exploit Q function 

		next_state=transition(state,a)#next state
		roundedstate=staterounding(state)
		roundednextstate=staterounding(next_state)
		
		if p.world[roundednextstate[0],roundednextstate[1]]==0 and next_state[0]<p.a and next_state[0]>0 and next_state[1]>0 and next_state[1]<p.b:	
			if np.linalg.norm(roundednextstate-target_state)<=p.thresh:
				R=p.highreward	#give high reward
				breakflag=1 #terminate episode
			else:
				R=p.livingpenalty # living penalty, if any
		else: 
			R=p.penalty # penalty if it bumps into an obstacle
			next_state=state.copy()# reset agent's state to what it was before bumping into the obstacle
		Qmaxnext, aoptnext=maxQ_tab(Q,next_state)
		Q[roundedstate[0],roundedstate[1],a]=Q[roundedstate[0],roundedstate[1],a]+p.alpha*(R+(p.gamma*Qmaxnext)-Q[roundedstate[0],roundedstate[1],a])#Q learning update

		state=next_state.copy()#update state
		roundedstate=staterounding(state)
		count=count+1
		if count>p.breakthresh:#terminate if max no of interactions has been reached
			breakflag=1
	return Q

def maxQ_tab(Q,state):
	#get max of Q values and corresponding action
	Qlist=[]
	roundedstate=staterounding(state)
	for i in range(p.A):
		Qlist.append(Q[roundedstate[0],roundedstate[1],i])
	tab_maxQ=np.max(Qlist)
	maxind=[]
	for j in range(len(Qlist)):
		if tab_maxQ==Qlist[j]:
			maxind.append(j)
	if len(maxind)>1:
		optact=maxind[np.random.randint(len(maxind))]
	else:
		optact=maxind[0]
	return tab_maxQ, optact

def transition(state,act):
	#print(orig_state)
	#print(act)
	n1 = np.random.uniform(low=-0.2, high=0.2, size=(1,))# x noise
	n2 = np.random.uniform(low=-0.2, high=0.2, size=(1,))# y noise
	new_state=state.copy()
	if act==0:
		new_state[0]=state[0]
		new_state[1]=state[1]+1#move up
	elif act==1:
		new_state[0]=state[0]+1#move right
		new_state[1]=state[1]
	elif act==2:
		new_state[0]=state[0]
		new_state[1]=state[1]-1#move down
	elif act==3:
		new_state[0]=state[0]-1#move left
		new_state[1]=state[1]

	new_state[0]=new_state[0]+n1
	new_state[1]=new_state[1]+n2
	return new_state

def calcret(Q,goal_state,episode_no):
	#This function just evaluates the performance of the agent at different stages of learning. 
	#The evaluation can also be done online, but this function makes multiple evaluations, giving a better estimate 
	ret=0
	eps_live=p.epsilon#1-(p.epsilon_decay*episode_no)
	for i in range(p.evalruns):
		breakflag=0
		initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
		rounded_initial_state=staterounding(initial_state)
		while p.world[rounded_initial_state[0],rounded_initial_state[1]]==1 or np.linalg.norm(rounded_initial_state-goal_state)<=p.thresh:
			initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
			rounded_initial_state=staterounding(initial_state)
		state=initial_state.copy()
		for j in range(p.evalsteps):
			if breakflag==1:
				break
			#if p.epsilon>np.random.sample():
			if eps_live>np.random.sample():
				#explore
				optact=np.random.randint(p.A)
			else:
				#exploit
				Qmaxopt,optact=maxQ_tab(Q,state)
			
			next_state=transition(state,optact)
			roundednextstate=staterounding(next_state)
			if p.world[roundednextstate[0],roundednextstate[1]]==0 and next_state[0]<p.a and next_state[0]>0 and next_state[1]>0 and next_state[1]<p.b:		
				if np.linalg.norm(roundednextstate-goal_state)<=p.thresh:
					R=p.highreward
					breakflag=1	
				else:
					R=p.livingpenalty
			else: 
				R=p.penalty
				next_state=state.copy()
			state=next_state.copy()
			#R=R*((p.gamma)**j)
			ret=ret+R*((p.gamma)**j)
	avgsumofrew=ret/p.evalruns
	return avgsumofrew



########Additional functions for visualization######
def plotmap(worldmap):
	#plots the obstacle map
	for i in range(p.a):
		for j in range(p.b):
			if worldmap[i,j]>0:
				plt.scatter(i,j,color='black')
	plt.show()

def staterounding(state):
	#rounds off states
	roundedstate=[0,0]
	roundedstate[0]=int(np.around(state[0]))
	roundedstate[1]=int(np.around(state[1]))
	if roundedstate[0]>=(p.a-1):
		roundedstate[0]=p.a-2
	elif roundedstate[0]<1:
		roundedstate[0]=1
	if roundedstate[1]>=(p.b-1):
		roundedstate[1]=p.b-2
	elif roundedstate[1]<=0:
		roundedstate[1]=1
	return roundedstate

def opt_pol(Q,state,goal_state):
	#shows optimal policy
	plt.figure(0)
	plt.ion()
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]>0:
				plt.scatter(i,j,color='black')
	plt.show()
	pol=[]
	statelog=[]
	count=1
	while np.linalg.norm(state-goal_state)>=1:
		Qm,a=maxQ_tab(Q,state)
		if np.random.sample()>0.9:
			a=np.random.randint(p.A)
		next_state=transition(state,a)
		roundednextstate=staterounding(next_state)
		if p.world[roundednextstate[0],roundednextstate[1]]==1:
			next_state=state.copy()
		pol.append(a)
		statelog.append(state)
		print(state)
		plt.ylim(0, p.b)
		plt.xlim(0, p.a)
		plt.scatter(state[0],state[1],(60-count*0.4),color='blue')
		plt.draw()
		plt.pause(0.1)
		state=next_state.copy()
		print(count)
		if count>=100:
			break
		count=count+1
	return statelog,pol

def mapQ(Q):
	#plots a map of the value function
	plt.figure(1)
	plt.ion
	Qmap=np.zeros((p.a,p.b))
	for i in range(p.a):
		for j in range(p.b):
 			Qav=0
 			for k in range(p.A):
 				Qav=Qav+Q[i,j,k]
 			Qmap[i,j]=Qav
	#Qfig=plt.imshow(np.rot90(Qmap))
	Qmap=Qmap-np.min(Qmap)
	if np.max(Qmap)>0:
		Qmap=Qmap/np.max(Qmap)
	plt.imshow(np.rot90(Qmap),cmap="gray")
	#plt.draw()
	plt.pause(0.0001)
	return Qmap

#######################################
if __name__=="__main__":
	#w,Qimall=Qlearn_main_vid()
	Q,retlog=Qlearn_multirun_tab()
	mr=(np.mean(retlog,axis=0))
	csr=[]
	for i in range(len(mr)):
		if i>0:			
			csr.append(np.sum(mr[0:i])/i)
	plt.figure(2)
	plt.plot(csr)
	plt.show()