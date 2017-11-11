#!/usr/bin/python
# script that approximates a paramaeter set for the PMT simulation at a local minimum
import matplotlib
import numpy as np
import glob,sys,time
import matplotlib.pyplot as plt
import pmt_sim
from operator import add
from functools import reduce
from scipy.optimize import leastsq
from bisect import bisect

DO_CALC=False

par=[0.017866911599288441, 0.11908805362830678, 0.743778516057228, 10306660.263163926]
par_err=[0.00010344778246309605, 0.0014226785241796408, 0.005179218251279624, 134058.02026587893]

def progress(x):
    sys.stdout.write('\r'+str(x))

sim_set=dict(pmt_sim.DEF_SET)

sim_set=dict(pmt_sim.DEF_SET)
sim_set['NOISE']=440428
sim_set['OFFSET']=0
sim_set['DYN_FACTORS']=pmt_sim.DYN_FACT_HAM
sim_set['N']=1e6*20
sim_set['GAIN']=1e7

sim_set['R_PE']=par[1]

# R_PE is a problem
X=np.linspace(-0e7,0.5e7,1000)

sim_set['DYN_EXP']=par[2]
sim_set['P_DSKIP']=par[0]

# function definitions
# draw and update a asci progress bar
def progress_bar(curr,goal=100,width=40,final=False):
    if not final:
        progress=max(int(np.ceil(curr/goal*width)),1)
        gen='['+('='*(progress-1))+'>'+(' '*(width-progress)+']')
        gen+=' %s/%s'%(repr(curr),repr(goal))
        if 'last' in progress_bar.__dict__:
            gen+=' ERT: %.2fm'%((time.time()-progress_bar.last)*(goal-curr)/60.)
    else:
        gen='['+('='*width)+']\n'
        if final==1:
            gen=' '*len(gen)+'\r'
        if 'last' in progress_bar.__dict__:
            del progress_bar.last

    progress_bar.last=time.time()
    sys.stdout.write('\r'+gen)
    sys.stdout.flush()

def calc_y(sim_set):
    alles,data=pmt_sim.pmt_sim_mt(sim_set,progess_cb=progress_bar,threads=4)
    data=[sorted(data[0]),sorted(reduce(add,data[1:],[]))]

    Y=[np.array(list(bisect(curr,x) for x in X))/len(curr) for curr in data]
    Y5=[bisect(curr,sim_set['NOISE']*5)/len(curr) for curr in data]
    Y[0]=1-Y[0]
    return Y,Y5

DO_CALC='--do-calc' in sys.argv
if DO_CALC:
    print('doing wiggle')
    wiggle=[('DYN_EXP',2),('P_DSKIP',0)]
    lst_y=[[],[]]
    lst_y5=[]
    for i in [-1,1]:
        for j in [-1,1]:
            for k in [-1,1]:
                sim_set['DYN_EXP']=par[2]+i*par_err[2]
                sim_set['P_DSKIP']=par[0]+j*par_err[0]
                sim_set['R_PE']=par[1]+k*par_err[1]
                tmp=calc_y(sim_set)
                print()
                for i in range(2):
                    lst_y[i].append(list(tmp[0][i]))
                lst_y5.append(tmp[1])

    print('main calc')
    sim_set['DYN_EXP']=par[2]
    sim_set['P_DSKIP']=par[0]
    sim_set['R_PE']=par[1]
    Y,Y5=calc_y(sim_set)
    print()

    print('lst_y5',lst_y5)
    print(np.array(lst_y).shape)
    print(np.array(list(zip(*lst_y))).shape)
    MM=[[np.array([evaluator(curr) for curr in zip(*(lst_y[i]))]) for evaluator in (min,max)] for i in range(2)]
    res=[[evaluator(list(zip(*lst_y5))[i]) for evaluator in (min,max)] for i in range(2)]
    print(res)

    data=[X,Y,MM,Y5,res]
    with open('out/cut_data.txt','w') as f:
        f.write(repr(data))
else:
    with open('out/cut_data.txt','r') as f:
        data=eval(f.read().replace('array',''))
    X,Y,MM,Y5,res=(np.array(i) for i in data)

X/=sim_set['GAIN']
fig=plt.figure(figsize=(10,7))
for i in range(2):
    plt.plot(X,Y[i]*1e2,zorder=100,label=['False positive','Missed real'][i])
    plt.fill_between(X,MM[i][1]*1e2,MM[i][0]*1e2,zorder=10,alpha=0.5)

the_point=(sim_set['NOISE']*5/sim_set['GAIN'],Y5[1]*1e2)
print(the_point)
plt.axvline(the_point[0],ymax=the_point[1]/20,c='k',zorder=200)
plt.axhline(the_point[1],xmax=the_point[0]/0.5,c='k',zorder=200)
plt.annotate('$5\sigma_{DRS}$',
        xy=(the_point[0]*1.05, the_point[1]*0.4),fontsize=27,zorder=4000,
        bbox={'fc':'w','alpha':0.5,'ec':'none','joinstyle':'bevel'}) # ,bbox={'fc':'w','ec':'k'}
plt.annotate('$R_{miss}=(%.2f\\pm %.2f)\\%%$'%(Y5[1]*1e2,(res[1][1]-res[1][0])/2*1e2),
        xy=(the_point[0]*0.05, the_point[1]*1.2),fontsize=26,zorder=4000,
        bbox={'fc':'w','alpha':0.5,'ec':'none','joinstyle':'bevel'}) # ,bbox={'fc':'w','ec':'k'}

plt.legend()
plt.ylim(0,20)
plt.xlim(min(X),max(X))
plt.xlabel('cut position relative to 1PE')
plt.ylabel('ratio in $[\%]$')
fig.tight_layout()
plt.savefig('out/cut.png')
plt.savefig('out/cut.pdf')

plt.show()
