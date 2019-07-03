import numpy as np
############
#world params
a=24#size of environment - x direction
b=21#size of environment - y direction
#######################
world=np.zeros((a,b))#obstacles
world[:,0]=1
world[0,:]=1
world[:,b-1]=1
world[a-1,:]=1
##########################

#horizontals:
world[1,4]=1
world[3:7,4]=1
world[8:15,4]=1
world[16:23,4]=1
world[1:4,7]=1
world[7:9,7]=1
world[10:17,7]=1
world[19:21,7]=1
world[22,7]=1
world[7:16,10]=1
world[1:4,11]=1
world[20:23,12]=1
world[7:10,13]=1
world[11:17,13]=1
world[1:4,16]=1
world[6:9,16]=1
world[10:12,16]=1
world[14:23,16]=1
#verticals
world[4,7:9]=1
world[4,10:12]=1
world[4,13:17]=1
world[5,1:4]=1
world[6,16:20]=1
world[7,7:13]=1
world[11,7:10]=1
world[12,1:4]=1
world[12,16:20]=1
world[16,9:13]=1
world[17,1]=1
world[17,3]=1
world[17,19]=1
world[17,17]=1
world[19,7:14]=1
world[19,15]=1
############################
#task1: 3,3
#task2: 3,18
#task3: 20,18
#task4: 20,3
#task5: 10,2
#task6: 11,11
#task7: 20,14
#task8: 2,8
#task9: 15,2
#task10: 12,8
#test: 8,8

targ=np.array([3.,3.])#target location
thresh=0.1#distance threshold
################
#Q learning params
Nruns=3#number of runs
alpha=0.05#learning rate
gamma=0.95#discound factor
epsilon=0.3#exploration parameter
episodes=2000#no. of episodes
A=4#no. of actions
highreward=1#reward for reaching good state
penalty=0.#-0.0001#reward for bumping into obstacle
livingpenalty=0#living reward, if any
breakthresh=100#max number of interactions per episode
evalruns=10#no. of evaluation runs for calcret function
evalsteps=100#no. of evaluation steps
epsilon_decay=0.0005
##############