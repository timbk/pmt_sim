"""
    Module containing the core PMT simulation
    provides the main simulaiton function pmt_sim(),
    their dependencies and some functions to evaluate the simulation result

    Author: Tim Kuhlbusch
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul,add
import bisect,itertools,time,sys
#from collections import Counter
from oldcounter import Counter
from multiprocessing import Pool

# sane matpllotlib default font
font = {'size'   : 27}
matplotlib.rc('font',**font)

# dynode voltages for the JUNO HAMAMATSU PMT
DYN_FACT_HAM=[2240,1210,1120,900,620,300,300,300,300,300]
# color palette for plots
COLORS=['blue','orange','green','yellow','red']
# default settings
DEF_SET={
        'N':4e5,            # total number of simulated events
        'P_DSKIP':0.015,    # propability for skipping a dynode
        'N_DYN':10,         # dynode count NOTE: is overwritten if DYN_FACT is not None
        'GAIN':1e7,         # total gain for the setup
        'R_PE':0.1,         # Rate of N_PE>0
        'GAUSS_MIN':50,     # minumum number of electrons to use gauss averaging for
        'NOISE':4.5e5,      # sampling noise
        'DYN_FACTORS':DYN_FACT_HAM, # dynode factors; None -> all dynodes equal
        'DYN_EXP':1.,       # exponent to get from voltage to secondary emission coefficient
        'INDIVIDUAL_PE':True,       # calculate all initial PEs on their own if set to True
        'INDIVIDUAL_SKIP':False,    # if set to false either all or no electrons skip a dynode
        'LATE_SKIPS':True,  # if set to False only allow skips that involve first dynode
        'COE':True,         # conservation of energy for dynode skipping
        'OFFSET':0,         # offset for the e- count
        'FIX_NPE':None,     # if not none set N_PE to this value instead of the poisson result
        }

def settings_lookup(set_dict,key):
    """
        return the setting for the given key from set_dict
        or if it is not set the default value from DEF_SET
    """
    if key=='N_DYN':
        return len(settings_lookup(set_dict,'DYN_FACTORS'))
    if key=='DYN_FACTORS' and (key not in set_dict or set_dict[key] is None):
        return [1. for i in range(set_dict['N_DYN'] if 'N_DYN' in set_dict else DEF_SET['N_DYN'])]
    if key in set_dict:
        return set_dict[key]
    return DEF_SET[key]

def norm_dfact(set_dict):
    """
        norm dynode factors
    """
    DYN_FACT=np.array(settings_lookup(set_dict,'DYN_FACTORS'))
    # apply secondary emmision exponen
    DYN_FACT=DYN_FACT**settings_lookup(set_dict,'DYN_EXP')
    DYN_FACT=DYN_FACT/sum(DYN_FACT)
    # normaleize for GAIN
    I_DYN_GAIN = DYN_FACT * (settings_lookup(set_dict,'GAIN') / reduce(mul,DYN_FACT,1)  )**(1/len(DYN_FACT))
    # print results TODO: maybe remove..
    return I_DYN_GAIN

def calc_first_dynode(P_DSKIP,N_DYN):
    """
        determine a random value for the first dynode that is hit
        this function assumes that the propability of skipping is
        the same for all dynodes
        P_DSKIP:    the propability of skipping a dynode
        N_DYN:      the amount of dynodes
        returns:    an index of a dynode
    """
    curr=0
    while curr<N_DYN:
        if np.random.random()<(1.-P_DSKIP):
            return curr
        curr+=1
    return curr

def dyn_mul(ie,dyn_gain,GAUSS_APPROX_MIN=50):
    """
        evaluate the secondary emission yield
        ie:                 number of incoming electrons
        dyn_gain:           gain for energy
                            of incoming electrons
        GAUSS_APPROX_MIN:   minimum to use gauss
                            instead of sum of poissons
    """
    if ie*dyn_gain>GAUSS_APPROX_MIN:
        # anwendung ZGS
        return int(np.random.normal(
                        ie*dyn_gain,
                        np.sqrt(ie*dyn_gain)
                        ))
    else:
        return sum(np.random.poisson(dyn_gain,size=ie))

def calc_pes_new(set_dict,I_NORM_DFACT=None,ne=1):
    """
        simulate the respnose of the PMT to the given number of PE
        set_dict:       the dict containing the simulation settings
        ne:             number of PEs
        I_NORM_DFACT:   is ignored and just there for backwards compatibility
        returns:        resulting electron count, skip count
    """
    # initialize variabels
    curr_dyn=0
    skips=0
    # HOF for gain factor calculation
    def gen_dyn_gain_func():
        b=settings_lookup(set_dict,'DYN_EXP')
        a=settings_lookup(set_dict,'GAIN')/float(reduce(mul,np.array(DYN_FACTORS)**b,1.))
        a**=1./len(DYN_FACTORS)
        return lambda x: a*x**b
    # HOF for skipping determination
    def gen_skip_func():
        # sc calculates the sip count
        if INDIVIDUAL_SKIP:
            def sc(ne):
                return 0 if ne<=0 else np.random.binomial(ne,P_DSKIP)
        else:
            def sc(ne):
                return ne if np.random.random()<P_DSKIP else 0
        if LATE_SKIPS:
            def f(ne,_dyn_idx,_E):
                return sc(ne)
        else:
            def f(ne,dyn_idx,E):
                if E==sum(DYN_FACTORS[:dyn_idx+1]):
                    return sc(ne)
                return 0
        return f
    # retrieve settings from set_dict
    GAUSS_MIN=settings_lookup(set_dict,'GAUSS_MIN')
    INDIVIDUAL_SKIP=settings_lookup(set_dict,'INDIVIDUAL_SKIP')
    LATE_SKIPS=settings_lookup(set_dict,'LATE_SKIPS')
    COE=settings_lookup(set_dict,'COE')
    P_DSKIP=settings_lookup(set_dict,'P_DSKIP')
    N_DYN=settings_lookup(set_dict,'N_DYN')
    DYN_FACTORS=settings_lookup(set_dict,'DYN_FACTORS')

    DYN_GAIN_FUNC=gen_dyn_gain_func()
    skip_count=gen_skip_func()
    # create Counter for electrons
    electrons=Counter({0:ne}) # start with electron with E=0
    # iterate over the dynodes
    while curr_dyn<N_DYN:
        # add energy from current dynode step to all electrons
        if COE:
            # die python2 version vom CIP kann leider keine dictionary comprehension ;(
            # electrons=Counter({E+DYN_FACTORS[curr_dyn]:electrons[E] for E in electrons})
            electrons=Counter(dict((E+DYN_FACTORS[curr_dyn],electrons[E]) for E in electrons))
        else:
            electrons=Counter({DYN_FACTORS[curr_dyn]:sum(electrons.values())})
        # go through the different Energy groups of the electrons and apply multiplications
        new_electrons=Counter()
        for E in electrons:
            n_skip=skip_count(electrons[E],curr_dyn,E)   # determin skipping count
            if not INDIVIDUAL_SKIP and n_skip>0:
                skips+=1
            new_electrons[E]+=n_skip                            # transfer skipping electrons
            new_electrons[0]+=dyn_mul(electrons[E]-n_skip,DYN_GAIN_FUNC(E),GAUSS_MIN)   # transfer multiplied electrons
        electrons=new_electrons
        # next dynode
        curr_dyn+=1
    return sum(electrons.values()),skips
    #return sum([electrons[E] for E in electrons]),skips # return number of created electrons

def single_run(set_dict):
    """
        helper function for the multi-threaded version of the simulation
        set_dict:   the settings dict
        returns:    resulting electron count, NPE/skip index
    """
    total_ne=0
    skips=0
    # determine PE count
    if settings_lookup(set_dict,'FIX_NPE') is None:
        n_pe=np.random.poisson(settings_lookup(set_dict,'R_PE'))
        # determine whether a signal is created
    else:
        n_pe=settings_lookup(set_dict,'FIX_NPE')
    curr_dyn=0
    #ne=n_pe
    if settings_lookup(set_dict,'INDIVIDUAL_PE'):
        for pe_idx in range(n_pe):
            ne,skips=calc_pes_new(set_dict,ne=1) # NOTE: uses NEW version
            total_ne+=ne
    else:
        total_ne,skips=calc_pes_new(set_dict,ne=n_pe) # NOTE: uses NEW version
    total_ne+=settings_lookup(set_dict,'OFFSET')
    # add noise
    total_ne+=int(np.round(np.random.normal(0,settings_lookup(set_dict,'NOISE'))))
    return total_ne,0 if n_pe==0 else (1 if n_pe>1 else min(1,skips)+2)

def pmt_sim(set_dict={},osf=1,threads=4,progess_cb=lambda x:None):
    """
        main simulation function with multithreadding

        set_dict:       the settings dictionary
        progess_cb:     is called to provide feedback on the progress
                        is called as progess_cb(perc) where perc is the progress in percent
        osf:            simulates N*osf events instead of N
                        must be corrected for in the evaluation of the results
        threads:        the number of threads for the created pool
                        as pnly one pool is created in the first call of this function,
                        this parameter is ignored at all subsequent calls
        returns:        simulation result with noise,
                        simulation result with noise individually
    """
    # test if the pool was already create or else create it
    if not ('P' in pmt_sim_mt.__dict__):
        pmt_sim_mt.P=Pool(threads)
    # fetch how many events are to be simulated
    N=int(settings_lookup(set_dict,'N'))*osf
    # generator function that just repeats the given item 'times' times
    def repeat(item,times):
        for i in range(times):
            yield item
    # the main "loop"
    ret=[]
    for rep in range(100):
        tmp=pmt_sim_mt.P.map(single_run,repeat(set_dict,int(N/100)))
        ret+=list(tmp)
        progess_cb(rep)
    per_dyn=list(list(map(lambda x: x[0],filter(lambda x: x[1]==i,ret))) for i in range(5))
    ret=list(map(lambda x: x[0],ret))
    return ret,per_dyn

# for backwards compatability
pmt_sim_mt=pmt_sim

def print_NPE_rates(per_dyn):
    """
        print N_PE counts
    """
    s=0
    for i in range(len(per_dyn)):
        print(i,len(per_dyn[i]))
        s+=len(per_dyn[i])
    print('total',s)

def gen_hist_data(data,nbin=120,rnge=(-5e6,3.5e7),osf=1):
    """
        calculate histograms for the simulation data
    """
    per_dyn_hist=[np.histogram(i,bins=nbin,range=rnge) for i in data]
    X=per_dyn_hist[0][1][:-1]
    D=X[1]-X[0]
    per_dyn_hist=[i[0]/osf for i in per_dyn_hist]
    YO=reduce(add,per_dyn_hist,np.zeros(len(per_dyn_hist[0])))
    return X,per_dyn_hist,YO,D

def gen_hist(fname,data,set_dict={}):
    """
        generate a histogram plot of the data
    """
    # generate histograms
    X,per_dyn_hist,YO,D=gen_hist_data(data)

    # plot results
    fig=plt.figure(figsize=(13,10))
    plt.bar(X,YO,D,log=True,color='w',edgecolor='k',zorder=-100,label='All')
    for idx in range(len(per_dyn_hist)):
        curr=per_dyn_hist[idx]
        if sum(curr)>0:
            plt.bar(X,curr,D,log=True,alpha=0.5,facecolor=COLORS[idx%len(COLORS)],
                    zorder=([1,20]+[10+i for i in range(settings_lookup(set_dict,'N_DYN'))])[idx],
                    label=(['0PE','NPE>1']+[('1PE; %i skip%s'%(i,'s' if i!=1 else '') if i<3 else None) for i in range(len(per_dyn_hist)-2)])[idx])
    # set settings for the plot
    plt.ylim(0.5,max(YO)*1.5)
    plt.xlim(-5e6,3.5e7)
    plt.legend()
    plt.xlabel('$N_{e^-}$')
    plt.ylabel('Count')
    if settings_lookup(set_dict,'NOISE')>0:
        plt.title('$N=%.1e$ $P_{skip}=%.3f$ $R_{PE}=%.3f$ $\sigma_{noise}=%.1e$'\
                %tuple(settings_lookup(set_dict,key) for key in ['N','P_DSKIP','R_PE','NOISE']))
    else:
        plt.title('Without noise $N=%.1e$ $P_{skip}=%.3f$ $R_{PE}=%.3f$'\
                %tuple(settings_lookup(set_dict,key) for key in ['N','P_DSKIP','R_PE']))
    plt.grid(ls=':')
    # save plot and clean up
    fig.tight_layout()
    plt.savefig(fname)
    plt.close(fig)

def gen_cut_ana(data,N=1000):
    """
        create the data for a percentage vs cut plot
    """
    ldat=[sorted(i) for i in [data[0],itertools.chain(*data[1:])]]
    cmax=max([i[-1] for i in ldat])
    cmin=min([i[0] for i in ldat])
    X=np.linspace(cmin,cmax,N)
    Y=[np.array([bisect.bisect(i,x) for x in X])/len(i) for i in ldat]
    Y[0]=1-Y[0]
    return X,Y

def progress_bar(curr,goal=100,width=40,final=False):
    """
        ASCI progress bar
    """
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
    print('\r'+gen,end='')
    sys.stdout.flush()
