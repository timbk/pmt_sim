import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob,sys,time
from operator import add,mul
from functools import reduce
# local imports
import pmt_sim
from helpers import read_data

# sane matpllotlib default font
font = {'size'   : 20}
matplotlib.rc('font',**font)
N_THREADS=4

# settings
OSF=1

# functions
def read_data(fname):
    data=np.genfromtxt(fname,delimiter=';')
    head=[i[0] for i in data[:8]]
    head[3]+=7
    head[4]+=7
    data=data[8:]
    #print(data)
    #print(data.shape)
    return head,data[:,0],data[:,1]       
def progress_bar(curr,goal=100,width=40,final=False):
    if not final:
        progress=max(int(np.ceil(curr/goal*width)),1)
        gen='['+('='*(progress-1))+'>'+(' '*(width-progress)+']')
        gen+=' %s/%s'%(repr(curr),repr(goal))
        if 'last' in progress_bar.__dict__:
            gen+=' ERT: %.2fm'%((time.time()-progress_bar.last)*(goal-curr)/60.)
    else:
        if final==1:
            gen=''
            gen='['+('='*width)+']\n'
            if 'last' in progress_bar.__dict__:
                del progress_bar.last
    progress_bar.last=time.time()
    sys.stdout.write('\r'+gen)
def gen_plot(f,outname,figsize=(14,10)):
    fig=plt.figure(figsize=figsize)
    ax=plt.gca()
    plt.grid(ls=':')
    f(ax)
    if int(sys.version.split('.')[0])>=3:
        fig.tight_layout()
    plt.savefig(outname+'.png')
    plt.savefig(outname+'.pdf')
    plt.close(fig)
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
# run the simulation with the given parameter set and calculate chi to the given data
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
        _,odata=pmt_sim.pmt_sim_mt(sset,osf=osf,threads=nthread,progess_cb=progress_bar)
        progress_bar(100,final=1)

        tmp=( np.histogram(np.array(curr),bins=int(hist_set['NBIN']),range=rnge) for curr in odata)
        per_dyn=(i[0] for i in tmp)
        bins=tmp[0][1][:-1]

        if full_hist is None:
            full_hist=per_dyn
        else:
            full_hist=list(np.array(arr+per_dyn[idx]) for idx,arr in enumerate(full_hist))
    Y=(reduce(add,full_hist,np.zeros(len(full_hist[0])))/OSF)[range(int(hist_set['SEL_LOW']),int(hist_set['SEL_HIGH']]
    assert len(Y)==len(cmp_data)
    chi=float(sum(((Y-cmp_data)/np.sqrt(cmp_data))**2)/(len(Y)-len(vpar)))
    if ret_res:
        per_dyn=[np.histogram(i,bins=int(hist_set['NBIN']),range=rnge)[0]/osf for i in odata]
        return chi,np.array(hist/osf),np.array(bins+(bins[1]-bins[0])/2.),list(full_hist),sset
    else:
        return chi


# load cmp file
cmp_head,cmp_X,cmp_Y=read_data('ref/data_1630_1.495_1500301284.csv')
#cmp_head,cmp_X,cmp_Y=read_data('sim_fit_data/data_1630_1.495_1500301284.txt')
NBIN=int(cmp_head[0])
BIN_RNGE=tuple(cmp_head[i] for i in range(1,3))
bin_sel=tuple(int(cmp_head[i]) for i in range(3,5))
print(BIN_RNGE)

# get parameters
sim_set=dict(pmt_sim.DEF_SET)
sim_set['NOISE']=cmp_head[6]
sim_set['OFFSET']=cmp_head[5]
best_fit=[  1.78996894e-02,  1.09890403e-01,  7.88473845e-01,  1.01845288e+07]
best_fit=[0.017866911599288441, 0.11908805362830678, 0.743778516057228, 10306660.263163926]
for i,key in enumerate(['P_DSKIP', 'R_PE', 'DYN_EXP', 'GAIN']):
    #sim_set[key]=float(input('Value for %s:\t'%key))
    sim_set[key]=best_fit[i]
    if key=='R_PE':
        sim_set['N']=cmp_head[7]*(1.+best_fit[i])
print(sim_set)

# plot result
print('Generating result plot')
chi_calc_set={'N':noise_fr[0][2],'NBIN':NBIN,'BIN_RNGE':tuple(BIN_RANGE),'SEL_LOW':start,'SEL_HIGH':end}
chi,Y,X,per_dyn,sset=calc_chi2(sim_set,best_fit,chi_calc_set,cmp_Y[start:end],osf=OSF*4,nthread=N_THREADS,ret_res=True)
def final_fit_plot(ax):
    #plt.plot(cX,gauss(noise_fr[0],cX),zorder=-10,ls='--')
    ax.errorbar(cmp_X,cmp_Y,np.sqrt(cmp_Y),ls='none',capsize=3,label='measurement',zorder=-10)
    for y,label in zip(per_dyn,['0PE','>1PE','1PE; 0 skips','1PE; 1 skip','1PE; >1 skips']):
        ax.step(X,y,where='mid',label=label,linestyle='-')
    ax.step(X,Y,where='mid',label='full sim result',c='k')
    for idx,i in enumerate([start,end-1]):
        plt.axvline(X[i],c='k',ls='--',label='border of the fit area' if idx==0 else '')
    plt.ylim(0.75,max(cmp_Y)*2.)
    plt.legend()
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
