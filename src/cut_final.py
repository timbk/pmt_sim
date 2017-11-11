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
from scipy.special import erf
from bisect import bisect

plt.xkcd()

DO_CALC=False
DO_FIX_PE='--fix-pe' in sys.argv

GAIN=1e7
SIGMA=0.1*GAIN
par=[0.017866911599288441, 0.11908805362830678, 0.743778516057228, 10306660.263163926]
par_err=[0.00010344778246309605, 0.0014226785241796408, 0.005179218251279624, 134058.02026587893]

def progress(x):
    sys.stdout.write('\r'+str(x))

sim_set=dict(pmt_sim.DEF_SET)

sim_set=dict(pmt_sim.DEF_SET)
sim_set['NOISE']=SIGMA
sim_set['OFFSET']=0
sim_set['DYN_FACTORS']=pmt_sim.DYN_FACT_HAM
sim_set['N']=1e6/10
sim_set['GAIN']=GAIN

sim_set['R_PE']=par[1]

# R_PE is a problem
X=np.linspace(-0e7,0.7e7,1000)

sim_set['DYN_EXP']=par[2]
sim_set['P_DSKIP']=par[0]

# function definitions
# draw and update a asci progress bar
def progress_bar(curr,goal=100,width=40,final=False,rep=None):
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

    if rep is not None:
        gen+=' rep #%u'%rep
    progress_bar.last=time.time()
    sys.stdout.write('\r'+gen)
    sys.stdout.flush()

def calc_y_memsave(sim_set,osf=1,rep=20):
    Y,Y5=[],[]
    for i in range(rep):
        y,y5=calc_y(sim_set,osf/rep)
        print(' part' + str(i+1) ,end='')
        Y.append(y)
        Y5.append(y5)
    Y=np.array(Y)
    Y=[reduce(add,[y[i] for y in Y],np.zeros(len(Y[0][i])))/rep for i in range(len(Y[0]))]
    Y5=[reduce(add,[y[i] for y in Y5],0)/rep for i in range(len(Y5[0]))]
    return Y,Y5

def calc_y(sim_set,osf=1):
    alles,data=pmt_sim.pmt_sim_mt(sim_set,progess_cb=progress_bar,threads=4,osf=osf)
    data=[sorted(data[0]),sorted(reduce(add,data[1:],[]))]

    Y=[np.array(list(bisect(curr,x) for x in X))/max(len(curr),1) for curr in data]
    Y5=[bisect(curr,SIGMA*5)/max(len(curr),1) for curr in data]
    Y[0]=1-Y[0]
    return Y,Y5

DO_CALC='--do-calc' in sys.argv
if DO_CALC:
    R_PE=[0.01,0.05,0.5,1,2,5]
    print('main calc')
    sim_set['DYN_EXP']=par[2]
    sim_set['P_DSKIP']=par[0]
    Y,Y5=[],[]
    for idx,r_pe in enumerate(R_PE):
        print('%u of %u'%(idx+1,len(R_PE)))
        sim_set['R_PE']=r_pe
        if DO_FIX_PE:
            sim_set['FIX_NPE']=idx+1
            if idx>2:
                break
        y,y5=calc_y_memsave(sim_set,osf=((5 if r_pe>0.05 else 20) if r_pe<=0.05 else 1) if r_pe<=1 else 0.2)
        Y.append(y)
        Y5.append(y5)
        print()

    data=[R_PE,X,Y,Y5]
    with open('out/cut_data_final'+('_fix_pe' if DO_FIX_PE else '')+'.txt','w') as f:
        f.write(repr(data))
else:
    print('using old data')
    with open('out/cut_data_final'+('_fix_pe' if DO_FIX_PE else '')+'.txt','r') as f:
        data=eval(f.read().replace('array',''))
    R_PE,X,Y,Y5=(np.array(i) for i in data)

X/=sim_set['GAIN']
fig=plt.figure(figsize=(10,7))
if DO_FIX_PE:
    plt.plot(X,(1-erf(X/(np.sqrt(2)*0.1)))/2*1e2,label='$N_{PE}=0$')
for idx,r_pe in enumerate(R_PE):
    if DO_FIX_PE:
        if idx>2:
            break
        label='$N_{PE}=%u$ (%s%%)'%\
                        (idx+1, 0 if Y5[idx][1]==0 else (('%.'+str(min(2,int(-np.log10(Y5[idx][1]*1e2)+2)))+'f')%(Y5[idx][1]*1e2)))
    else:
        label='$R_{PE}=%s$ (%s%%)'%\
                        (('%.2f' if r_pe<1 else '%.0f')%(r_pe), ('%.2f' if r_pe>2 else ('%.0f' if r_pe<1 else '%.1f'))%(Y5[idx][1]*1e2))
    plt.plot(X,Y[idx][1]*1e2,zorder=-1,label=label)
    the_point=(sim_set['NOISE']*5/sim_set['GAIN'],Y5[idx][1]*1e2)
    print(the_point)
    plt.axhline(the_point[1],xmax=the_point[0]/0.7,c='k',ls='--')
    if idx==0:
        plt.axvline(the_point[0],ymax=the_point[1]/20,c='k',ls='--')

plt.legend(fontsize=20)
plt.ylim(0,20)
plt.xlim(min(X),max(X))
plt.xlabel('cut position relative to 1PE')
plt.ylabel('missed ratio in $[\%]$')
fig.tight_layout()
plt.savefig('out/cut_final'+('_fix_pe' if DO_FIX_PE else '')+'.png')
plt.savefig('out/cut_final'+('_fix_pe' if DO_FIX_PE else '')+'.pdf')

plt.show()
