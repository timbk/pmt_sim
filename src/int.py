#!/usr/bin/python3
"""
    interactive simulation of the 20" Hamamatsu PMT for JUNO
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob,sys
from matplotlib.widgets import Slider,RadioButtons,Button,CheckButtons
import tkinter as tk
from tkinter import filedialog,messagebox
# local imports
import pmt_sim as sim
from helpers import read_data

# general settings
NEW_PLOT=True # new plot style (steps)
N_THREAD=4
NBIN=200
# replace the yellow of the old color palette as it does not work well on white bg
if NEW_PLOT:
    sim.COLORS[3]='purple'

# gloabel matplotlib settings
font = {'size'   : 20}
matplotlib.rc('font',**font)
matplotlib.rcParams['errorbar.capsize'] = 3

# start up tkinter
tk_root=tk.Tk()
tk_root.withdraw()

# mutex to have only one updater running
updater_mutex=False

# create settings dict
CURR_SET=dict(sim.DEF_SET)
CURR_SET['DYN_FACTORS']=sim.DYN_FACT_HAM
print(np.array(sim.DYN_FACT_HAM)/sum(sim.DYN_FACT_HAM))
CURR_SET['N']=1e5

# calculate initial result
pcb=lambda x: sys.stdout.write('\r%u'%x);sys.stdout.flush()
_,data=sim.pmt_sim_mt(CURR_SET,progess_cb=pcb,threads=N_THREAD)
print()
X,per_dyn_hist,YO,D=sim.gen_hist_data(data)
CURR_RES=data

# set up cut diagram
CUT_DATA=sim.gen_cut_ana(data)
cut_fig,cut_ax=plt.subplots()
CUT_PLOTS=(
    cut_ax.plot(CUT_DATA[0],CUT_DATA[1][0]*100,label='false positive')[0],
    cut_ax.plot(CUT_DATA[0],CUT_DATA[1][1]*100,label='missed real')[0],
    )
plt.xlim(CUT_DATA[0][0],1.7e7)
plt.xlabel('cut position')
plt.ylabel('ratio in $[\\%]$')
plt.grid(ls=':')
plt.legend()

# set up figure and plots
fig,ax=plt.subplots()
plt.subplots_adjust(bottom=0.35,top=0.95)
ax.set_yscale('log')
ax.grid(ls=':')
CURR_REF_PLOT=None
CURR_REF_DATA=None
CUT_VALUE=CURR_SET['NOISE']*5

# setup barplots
if NEW_PLOT:
    TOT_BAR=plt.step(X+D,YO,c='grey',zorder=1e6,label='All',linewidth=2,linestyle='-')
else:
    TOT_BAR=plt.bar(X+D/2,YO,D,log=True,color='w',edgecolor='k',zorder=-100,label='All')
PBARS=[]
for idx in range(len(per_dyn_hist)):
    curr=per_dyn_hist[idx]
    if NEW_PLOT:
        PBARS.append(plt.step(X+D,curr,c=sim.COLORS[idx%len(sim.COLORS)],linewidth=2,
                linestyle='-' if idx<len(sim.COLORS) else ':',
                zorder=([1,20]+[10+i for i in range(len(per_dyn_hist))])[idx],
                label=(['0PE','NPE>1']+[('1PE; %i skip%s'%(i,'s' if i!=1 else '') if i<3 else None) for i in range(len(per_dyn_hist)-2)])[idx]))
    else:
        if sum(curr)>0:
            PBARS.append(plt.bar(X+D/2,curr,D,log=True,alpha=0.5,facecolor=sim.COLORS[idx%len(sim.COLORS)],
                    zorder=([1,20]+[10+i for i in range(len(per_dyn_hist))])[idx],
                    label=(['0PE','NPE>1']+[('1PE; %i skip%s'%(i,'s' if i!=1 else '') if i<3 else None) for i in range(len(per_dyn_hist)-2)])[idx]))
plt.legend(fontsize=13)
plt.xlabel('$N_{e^-}$')
plt.ylabel('Count')
CUT_RES_TEXT=plt.annotate('test',xy=(0.9,0.4),zorder=100000,xycoords='axes fraction',bbox={'fc':'w','ec':'k'})

## create GUI
# create cut slier
CUT_VIS=plt.axvline(CUT_VALUE,c='k',ls='--')
def update_cut_val():
    global CUT_VIS,CUT_VALUE
    CUT_VIS.set_xdata([CUT_VALUE+CURR_SET['OFFSET'] for it in CUT_VIS.get_xdata()])
    false_positive=len(list(filter(lambda x: x>CUT_VALUE+CURR_SET['OFFSET'],CURR_RES[0])))/len(CURR_RES[0])
    missed=sum([len(list(filter(lambda x: x<CUT_VALUE+CURR_SET['OFFSET'],i))) for i in CURR_RES[1:]]) \
                        /sum([len(i) for i in  CURR_RES[1:]])
    CUT_RES_TEXT.set_text('$R_{FP}=%.3f\\%%$\n$R_{missed}=%.3f\\%%$'%(false_positive*100.,missed*100))

SLIDER_CUT=Slider(plt.axes([0.1,0.96,0.7,0.03]),'cut',-1e7,2e7,valinit=CUT_VALUE,valfmt='%.2e')
def cut_slider_handler(val):
    global CUT_VALUE
    CUT_VALUE=val
    update_cut_val()
SLIDER_CUT.on_changed(cut_slider_handler)
update_cut_val()

# create update button
BUTTON_GEN=Button(plt.axes([0.8,0.07,0.15,0.05]),'Generate')
STATUS_TEXT=plt.text(0.0,-1.2,'Done',backgroundcolor='w')
def update(event,osf=1):  # GenerateButton click handler ; run the simulation once
    global updater_mutex,TOT_BAR
    if updater_mutex:
        return
    STATUS_TEXT.set_text('Running..')
    ax.draw_artist(STATUS_TEXT)
    ax.figure.canvas.blit(STATUS_TEXT.get_window_extent())
    def updater_cb(n):
        STATUS_TEXT.set_text('Running.. (%u%%)'%(n))
        ax.draw_artist(STATUS_TEXT)
        ax.figure.canvas.blit(STATUS_TEXT.get_window_extent())
    updater_mutex=True
    _,data=sim.pmt_sim_mt(CURR_SET,osf=osf,progess_cb=updater_cb)
    CURR_RES=data
    #for idx,curr in enumerate(data):
    #    print(idx,len(curr))
    X,per_dyn_hist,YO,D=sim.gen_hist_data(data,osf=osf,nbin=NBIN)
    # update individual plots
    if NEW_PLOT:
        TOT_BAR[0].remove()
        TOT_BAR=ax.step(X+D,YO,c='grey',zorder=1e6,label='All',linewidth=2,linestyle='-')
        for idx,curr in enumerate(per_dyn_hist):
            # remove old plot
            PBARS[idx][0].remove()
            # create new plot
            PBARS[idx]=ax.step(X+D,curr,c=sim.COLORS[idx%len(sim.COLORS)],linewidth=2,
                    linestyle='-' if idx<len(sim.COLORS) else ':',
                    zorder=([1,20]+[10+i for i in range(len(per_dyn_hist))])[idx],
                    label=['0PE','>1PE','1PE; 0skips','1PE; 1skip','1PE; >1skip'][idx])
    else:
        # update sum plot
        for b,val in zip(TOT_BAR,YO):
            b.set_height(val)
        # update plot
        for curr in range(len(PBARS)):
            for b,val in zip(PBARS[curr],per_dyn_hist[curr]):
                b.set_height(val)

    update_cut_val()
    # update cut plot
    CUT_DATA=sim.gen_cut_ana(data)
    for i in range(2):
        CUT_PLOTS[i].set_ydata(CUT_DATA[1][i]*100.)
        CUT_PLOTS[i].set_xdata(CUT_DATA[0]-CURR_SET['OFFSET'])
    cut_fig.canvas.draw()

    print('updater finished')
    sim.print_NPE_rates(data)
    STATUS_TEXT.set_text('Done')
    plt.draw()
    updater_mutex=False

BUTTON_GEN.on_clicked(update)

# 'schoen machen' button
BUTTON_SCHOEN=Button(plt.axes([0.8,0.19,0.15,0.05]),'Schoen')
BUTTON_SCHOEN.on_clicked(lambda x: update(x,osf=20))

# load refference button
BUTTON_LREF=Button(plt.axes([0.8,0.13,0.18,0.05]),'Load Measurement')
BUTTON_LREF.label.set_fontsize(16)
def button_load_ref(event): # handler for BUTTON_LREF clicked
    global CURR_REF_DATA,CURR_REF_PLOT
    is_tia=messagebox.askquestion('TIA?','Do you want to load a TIA measurement?')
    fname=filedialog.askopenfilename()
    print(fname)
    if type(fname) is str:
        CURR_REF_DATA,extra_info,_,soffset=read_data(fname)
        print(is_tia)
        if is_tia=='yes':
            CURR_REF_DATA=np.array(CURR_REF_DATA)/2.
        CURR_REF_DATA=np.histogram(CURR_REF_DATA,bins=NBIN,range=(-5e6,3.5e7))

        if CURR_REF_PLOT is not None:
            CURR_REF_PLOT.remove()
            del CURR_REF_PLOT
        print(CURR_REF_DATA[0])
        rD=CURR_REF_DATA[1][1]-CURR_REF_DATA[1][0]
        CURR_REF_PLOT=ax.errorbar(CURR_REF_DATA[1][:-1]+rD/2,CURR_REF_DATA[0],np.sqrt(CURR_REF_DATA[0]),
                ls='none',c='k',label='Refference',zorder=1000000)
        plt.legend()

BUTTON_LREF.on_clicked(button_load_ref)

# create sliders
def make_slider_handler(key): # slider value change handler
    def slider_handler(val):
        print('changing',key)
        CURR_SET[key]=val
    return slider_handler

SLIDER_SET=[
        ('N',1e4,10e5,'%.2e'),
        ('OFFSET',-1e6,1e6,'%.2e'),
        ('P_DSKIP',0,0.2,'%.3f'),
        ('R_PE',0,0.3,'%.3f'),
        ('GAIN',5e6,5e7,'%.2e'),
        ('GAUSS_MIN',0,100,'%.1f'),
        ('NOISE',0,1e6,'%.2e'),
        ('DYN_EXP',0,2,'%.2f'),
        ]
SLIDER_AXES=[plt.axes([0.14,0.22-i*0.03,0.5,0.02]) for i in range(len(SLIDER_SET))]
SLIDERS=[Slider(axes,curr[0],curr[1],curr[2],valinit=CURR_SET[curr[0]],valfmt=curr[3]) \
        for curr,axes in zip(SLIDER_SET,SLIDER_AXES)]
for curr,attr in zip(SLIDERS,SLIDER_SET):
    curr.on_changed(make_slider_handler(attr[0]))
    curr.valtext.set_fontsize(16)
    curr.label.set_fontsize(16)

# create check buttons
CHECK_BOX_LABELS=('idvdl PE','idvdl skip','late skips','CoE')
CHECK_BOX_SETTINGS=('INDIVIDUAL_PE','INDIVIDUAL_SKIP','LATE_SKIPS','COE')
CHECK_BOXES_AX=plt.axes([0.85,0.35,0.15,0.20])
CHECK_BOX_STATUS=[True,False,True,True]
CHECK_BOXES=CheckButtons(CHECK_BOXES_AX,CHECK_BOX_LABELS,CHECK_BOX_STATUS)
def check_box_handler(event):
    for label,key in zip(CHECK_BOX_LABELS,CHECK_BOX_SETTINGS):
        if event==label:
            idx=CHECK_BOX_LABELS.index(label)
            CHECK_BOX_STATUS[idx]=not CHECK_BOX_STATUS[idx]
            CURR_SET[key]=CHECK_BOX_STATUS[idx]

CHECK_BOXES.on_clicked(check_box_handler)

# run
plt.show()
