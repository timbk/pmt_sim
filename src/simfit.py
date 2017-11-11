#!/usr/bin/python
# script that approximates a paramaeter set for the PMT simulation at a local minimum
import matplotlib
matplotlib.use('Agg')
import numpy as np
import glob,sys,time
import matplotlib.pyplot as plt
import pmt_sim
from operator import add
from functools import reduce
from helpers import read_data
from scipy.optimize import leastsq

# settings
NBIN=200
BIN_RANGE=(-0.5e7,3.5e7)
D=(BIN_RANGE[1]-BIN_RANGE[0])/NBIN
MAX_GD_STEPS=0 # NOTE: set to 60 later
SIM_TOGGLES={'COE':True,'INDIVIDUAL_SKIP':True,'LATE_SKIPS':False}

# options for the chi^2 calculations
N_THREADS=4
OSF=100 # TODO: set OSF
#REP_SAMPE=1

# definitions
real_gauss=lambda p,x: np.exp( -(x-p[0])**2/p[1]**2/2. )/np.sqrt(2*np.pi)/p[1] * p[2] # 0:mu 1:sig 2:N
gauss_d=float(D)
gauss=lambda p,x: real_gauss(p,x)*gauss_d
IV_FORMAT=['P_DSKIP','R_PE','DYN_EXP','GAIN']

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

# make strings with an underscore better readable as latex code
def make_tex(s):
    s=s.split('_')
    if len(s)==0:
        return s[0]
    return s[0]+'_{%s}'%(''.join(s[1:]))

# decorator to reapeatedly run a function and calculate its outputs mean and std
def run_avrg(n):
    def ret(f):
        def new_f(*arg,**kwarg):
            lst=[]
            for i in range(n):
                lst.append(f(*arg,**kwarg))
            return np.mean(lst),np.std(lst,ddof=1) if n>1 else 0#/np.sqrt(n)
        return new_f
    return ret

# decorator to time the
def run_timing(f):
    def new_f(*arg,**kwarg):
        start=time.time()
        ret=f(*arg,**kwarg)
        print('calc took %.1fm'%((time.time()-start)/60.))
        return ret
    return new_f

@run_timing
# run the simulation with the given parameter set and calculate chi to the given data
def calc_chi2_nosplit(sset,vpar,hist_set,cmp_data,ret_res=False,osf=OSF,nthread=N_THREADS):
    # set the new parameters in the simulation settings
    for val,key in zip(vpar,IV_FORMAT):
        sset[key]=val
    # set n
    sset['N']=hist_set['N']*(1+vpar[1])
    #print(sset)
    # run simulation
    if nthread>1:
        #print(sset,OSF,nthread)
        data,odata=pmt_sim.pmt_sim_mt(sset,osf=osf,threads=nthread,progess_cb=progress_bar)
    else:
        odata,_,_=pmt_sim.pmt_sim(sset,osf=osf,progess_cb=progress_bar)
        data=reduce(add,odata,[])
    progress_bar(100,final=1)
    # calculate chi
    rnge=tuple(hist_set['BIN_RNGE'])
    hist,bins=np.histogram(np.array(data),bins=int(hist_set['NBIN']),range=rnge)
    Y=np.array(hist[range(int(hist_set['SEL_LOW']),int(hist_set['SEL_HIGH']))])/osf
    assert len(Y)==len(cmp_data)
    if ret_res:
        per_dyn=[np.histogram(i,bins=int(hist_set['NBIN']),range=rnge)[0]/osf for i in odata]
        return float(sum(((Y-cmp_data)/np.sqrt(cmp_data))**2)/(len(Y)-len(vpar))),\
                np.array(hist/osf),np.array(bins[:-1]+(bins[1]-bins[0])/2.),list(per_dyn),sset
    else:
        return float(sum(((Y-cmp_data)/np.sqrt(cmp_data))**2)/(len(Y)-len(vpar)))
# run the simulation with the given parameter set and calculate chi to the given data
@run_timing
def calc_chi2(sset,vpar,hist_set,cmp_data,ret_res=False,osf=OSF,nthread=N_THREADS,nsplit=20):
    # set the new parameters in the simulation settings
    for val,key in zip(vpar,IV_FORMAT):
        sset[key]=val
    # set n
    sset['N']=hist_set['N']*(1+vpar[1])
    rnge=tuple(hist_set['BIN_RNGE'])

    full_hist=None
    for rep in range(nsplit):
        print('split %u of %u'%(rep+1,nsplit))
        _,odata=pmt_sim.pmt_sim_mt(sset,osf=osf/nsplit,threads=nthread,progess_cb=progress_bar)
        progress_bar(100,final=1)

        tmp=list( np.histogram(np.array(curr),bins=int(hist_set['NBIN']),range=rnge) for curr in odata)
        per_dyn=list(i[0] for i in tmp)
        bins=tmp[0][1][:-1]

        if full_hist is None:
            full_hist=per_dyn
        else:
            full_hist=list(np.array(arr+per_dyn[idx]) for idx,arr in enumerate(full_hist))
    full_hist=list(i/osf for i in full_hist)
    Y=(reduce(add,full_hist,np.zeros(len(full_hist[0]))))[range(int(hist_set['SEL_LOW']),int(hist_set['SEL_HIGH']))]
    assert len(Y)==len(cmp_data)
    chi=float(sum(((Y-cmp_data)/np.sqrt(cmp_data))**2)/(len(Y)-len(vpar)))
    if ret_res:
        per_dyn=[np.histogram(i,bins=int(hist_set['NBIN']),range=rnge)[0]/osf for i in odata]
        return chi,np.array(reduce(add,full_hist,np.zeros(len(full_hist[0])))),np.array(bins+(bins[1]-bins[0])/2.),list(full_hist),sset
    else:
        return chi

# generate a plot image
def gen_plot(f,outname,figsize=(10,7)):
    fig=plt.figure(figsize=figsize)
    ax=plt.gca()
    plt.grid(ls=':')
    f(ax)
    if int(sys.version.split('.')[0])>=3:
        fig.tight_layout()
    plt.savefig(outname+'.png')
    plt.savefig(outname+'.pdf')
    plt.close(fig)

# generate a plot image for the standard output plot
def gen_std_plot(f,outname,figsize=None):
    def tmp(ax):
        ax.set_yscale('log')
        f(ax)
        plt.xlabel('$N_{e^-}$')
        plt.ylabel('Event count')
    if figsize is not None:
        gen_plot(tmp,outname,figsize=figsize)
    else:
        gen_plot(tmp,outname)

# HOF that generates an itemgetter
def itemgetter(idx):
    return lambda x: x[idx]

USAGE='usage: ./simfit.py file.csv (P_DSKIP,R_PE,DYN_EXP,GAIN)\n\n\
        \tfile.csv: file with the measured data\n\
        \t(P_DSKIP,R_PE,DYN_EXP,GAIN): initial fit values'

# parse parameters
IS_TIA='--tia' in sys.argv
sys.argv=[i for i in sys.argv if i[0]!='-']
if len(sys.argv)!=3:
    print(USAGE)
    exit(1)
print('sys.argv',sys.argv)
fname=sys.argv[1]
iv=eval(sys.argv[2])
assert type(iv)==tuple
print('paramter format: %s'%repr(IV_FORMAT))
print('using %s as iv'%repr(iv))

# read measurement data
cmp_data,extra_info,_,soffset=read_data(fname)
cmp_data=np.array(cmp_data)
if IS_TIA:
    cmp_data/=2.

# do hist of meas. data
hist,bins=np.histogram(cmp_data,bins=NBIN,range=BIN_RANGE)
cmp_Y=hist
D=bins[1]-bins[0]
cmp_X=bins[:-1]+D/2.

fname=('.'.join(fname.split('.')[:-1])).split('/')[-1]

# do the noise fit
sel=cmp_Y>(max(cmp_Y)/10.)
noise_fit_X=cmp_X[sel]
noise_fit_Y=cmp_Y[sel]
noise_chi=lambda p,x,y,ey: (y-gauss(p,x))/ey
noise_iv=(cmp_X[list(cmp_Y).index(max(cmp_Y))],max(noise_fit_X)-min(noise_fit_X),len(cmp_data)*0.9)
print('noise_iv',noise_iv)
fr=leastsq(noise_chi,noise_iv,(noise_fit_X,noise_fit_Y,np.sqrt(noise_fit_Y)),full_output=True)
noise_fr=(fr[0],tuple(fr[1][i][i] for i in range(len(fr[0]))),sum(fr[2]['fvec']**2),len(noise_fit_X)-len(noise_iv))
print('noise_fr',noise_fr)
print('N_cmp_data',len(cmp_data))
noise_X_diff=max(noise_fit_X)-min(noise_fit_X)
cX=np.linspace(min(noise_fit_X)-noise_X_diff/2,max(noise_fit_X)+noise_X_diff/2,500)

# select the main fit range
start=sel.argmax()
end=((cmp_X<=cmp_X[start])|(cmp_Y>2)).argmin()
print('selected area for full fit',start,end,'of',len(cmp_Y))

# generate plot for the noise fit
def noise_fit_plot(ax):
    plt.plot(cX,gauss(noise_fr[0],cX),zorder=-10,ls='--')
    ax.errorbar(cmp_X,cmp_Y,np.sqrt(cmp_Y),ls='none')
    ax.errorbar(cmp_X[start:end],cmp_Y[start:end],np.sqrt(cmp_Y[start:end]),ls='none')
    ax.errorbar(cmp_X[sel],cmp_Y[sel],np.sqrt(cmp_Y[sel]),ls='none',capsize=3)
    plt.ylim(0.75,max(cmp_Y)*2.)
gen_std_plot(noise_fit_plot,'out/%s_noise'%fname)

# do the actual fit
# create simulation settings dictionary
sim_set=dict(pmt_sim.DEF_SET)
sim_set['NOISE']=noise_fr[0][1]
sim_set['OFFSET']=noise_fr[0][0]
sim_set['DYN_FACTORS']=pmt_sim.DYN_FACT_HAM
for key in SIM_TOGGLES:
        sim_set[key]=SIM_TOGGLES[key]
N_NOISE=noise_fr[0][2]
# graident descent settings
gf=np.array([0.8e-4,0.8e-4,0.8e-4,0.8e-4])
step_factor=2e-3
par=tuple(iv)
step=np.array(iv)*step_factor
chi_calc_set={'N':noise_fr[0][2],'NBIN':NBIN,'BIN_RNGE':tuple(BIN_RANGE),'SEL_LOW':start,'SEL_HIGH':end}
chi2=lambda x: calc_chi2(sim_set,x,chi_calc_set,cmp_Y[start:end])

# main fit loop
cnt=0
print('starting gradient descent')
STEPS=[]
while cnt<MAX_GD_STEPS:
    # calculate chi for current parameters
    ochi2=chi2(par)
    print('iteration',cnt)
    print('chi^2',ochi2)
    print('with par:',repr(par))
    STEPS.append(tuple((ochi2,))+tuple(par))
    # test if done
    if cnt>10:
        # end if the relative difference between
        stds=(np.std(list(map(lambda x: x[0],STEPS[-10:-5]))),
                np.std(list(map(lambda x: x[0],STEPS[-5:]))))
        print('testing brake with stds:',stds)
        if stds[1]/stds[0]<1.1:
            break
    grad=[]
    print('calculating gradient')
    for i in range(len(iv)):
        tmp=list(par)
        tmp[i]+=step[i]
        nchi2=chi2(tmp)
        grad.append((ochi2-nchi2)/step_factor)
    print('gradient',grad)
    print('diff',np.array(gf)*np.array(grad))
    par=tuple(par[i]+gf[i]*grad[i]*iv[i] for i in range(len(par)))
    print('new par:',repr(par))
    cnt+=1
print('Gradient descent finished after %u iterations'%cnt)

# determine fit result
# NOTE: not tested yet
CHIS=list(map(itemgetter(0),STEPS))
PARS=[list(map(itemgetter(idx+1),STEPS)) for idx in range(len(par))]
if len(STEPS)>=10:
    RES_START=-5
    fchi=np.mean(CHIS[RES_START:])
    fchi_err=np.std(CHIS[RES_START:],ddof=1)
    fpar=[np.mean(curr[RES_START:]) for curr in PARS]
    fpar_err=[np.std(curr[RES_START:],ddof=1) for curr in PARS]
    gd_res=fpar
    # print fit results
    print('chi',fchi,fchi_err)
    print('par',fpar,fpar_err)
    with open('out/res.log','a+') as f:
        f.write(fname+'\n') 
        f.write('fchi '+repr(fchi)+'\n')
        f.write('fchi_err '+repr(fchi_err)+'\n')
        f.write('fpar '+repr(fpar)+'\n')
        f.write('fpar_err '+repr(fpar_err)+'\n')
else:
    gd_res=par

# plot graph of the gradient descent
print('generating process plot')
def process_plot(ax):
    # upper plot with chi^2 data
    ax=plt.subplot(211)
    plt.plot(CHIS)
    plt.ylabel('$\\chi^2/ndof.$')
    # lower plot with the parameter values
    ax=plt.subplot(212)
    for curr,label in zip([np.array(j)/iv[idx] for idx,j in enumerate(PARS)],map(make_tex,IV_FORMAT)):
        plt.plot(curr,label='$%s$'%label)
    plt.legend()
    plt.ylabel('relative param. value')
    plt.xlabel('iteration')
gen_plot(process_plot,'out/%s_proc'%fname,figsize=(14,10))

# plot result
print('Generating result plot')
chi,Y,X,per_dyn,sset=calc_chi2(sim_set,gd_res,chi_calc_set,cmp_Y[start:end],osf=OSF*10,nthread=N_THREADS,ret_res=True)
def final_fit_plot(ax):
    #plt.plot(cX,gauss(noise_fr[0],cX),zorder=-10,ls='--')
    ax.errorbar(cmp_X,cmp_Y,np.sqrt(cmp_Y),ls='none',capsize=3,label='measurement',zorder=-10)
    for y,label in zip(per_dyn,['0PE','>1PE','1PE; 0 skips','1PE; 1 skip','1PE; >1 skips']):
        ax.step(X,y,where='mid',label=label,linestyle='-')
    ax.step(X,Y,where='mid',label='full sim result',c='k')
    for idx,i in enumerate([start,end-1]):
        plt.axvline(X[i],c='k',ls='--',label='border of the fit area' if idx==0 else '')
    plt.ylim(0.75,max(cmp_Y)*2.)
    #plt.legend()
    # TODO: add error to the paramater values
    atext='$\\chi^2/ndof=%.3f$\n'%(chi)+\
            '\n'.join(['$%s=%.2e$'%(make_tex(key),sset[key]) for key in ['N','NOISE','GAIN','R_PE','P_DSKIP','DYN_EXP']])
    plt.annotate('\n'.join(atext.split('\n')[:3]),
            xy=(0.2, 0.8), xycoords='axes fraction',fontsize=20,bbox={'fc':'w','ec':'k'},zorder=4000)
    plt.annotate('\n'.join(atext.split('\n')[3:]),
            xy=(0.45, 0.75), xycoords='axes fraction',fontsize=20,bbox={'fc':'w','ec':'k'},zorder=4000)
gen_std_plot(final_fit_plot,'out/%s_final'%fname,figsize=(14,7))
# generate residual plot
def final_fit_plot_res(ax):
    dY=Y-cmp_Y
    ax.errorbar(cmp_X,dY/np.sqrt(cmp_Y),1.,ls='none',capsize=3,label='residuen')
    plt.axhline(0,c='k')
    for idx,i in enumerate([start,end-1]):
        plt.axvline(X[i],c='k',ls='--',label='border of the fit area' if idx==0 else '')
    plt.xlabel('$N_{e^-}$')
    plt.ylabel('Residuals $[1]$')
gen_plot(final_fit_plot_res,'out/%s_resi'%fname,figsize=(14,7))
