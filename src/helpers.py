# version without matplotlib settings
import matplotlib
import numpy as np
from bisect import bisect_left
from scipy.optimize import leastsq,fmin
import matplotlib.pyplot as plt

# settings
SIGMA_CUT=0.6

# definitions
AVRG_MSTRING="#AVG_WV:[" # key string for parsing
AVRGPK_MSTRING="#AVG_PEAK:[" # key string for parsing
SPIKED_MSTRING="#SPIKED:" # key string for parsing
R=50. # resistance to calculate electron count
e=1.602176620898e-19

def smartbar(left,*args,**wargs):
    """
        automatically correct for the change in the usage of the left argument in different matplotlib versions
    """
    if matplotlib.__version__[0]=='2':
        if 'width' in wargs:
            plt.bar(left+wargs['width']/2.,*args,**wargs)
        else:
            plt.bar(left+args[1]/2.,*args,**wargs)
    else:
        plt.bar(left,*args,**wargs)

def read_data(fname,sigma_cut=SIGMA_CUT):
    """
        function that loads the data from a given file
    """
    global R
    charge=[]
    soffset=[]
    extra_info=dict()
    avrg_wv=None; avrgpk_wv=None
    f=open(fname,'r')
    for line in f:
        # handle extra information
        if line[0]=='#':
            if len(line)>len(AVRG_MSTRING) and line[:len(AVRG_MSTRING)]==AVRG_MSTRING:
                line=line[len(AVRG_MSTRING):]
                line=[j for j in line.split(']')[0].split('|') if len(j)>0]
                #print(line)
                line=[float(j) for j in line]
                avrg_wv=np.array(line)
                continue
            elif len(line)>len(AVRGPK_MSTRING) and line[:len(AVRGPK_MSTRING)]==AVRGPK_MSTRING:
                line=line[len(AVRGPK_MSTRING):]
                line=[j for j in line.split(']')[0].split('|') if len(j)>0]
                #print(line)
                line=[float(j) for j in line]
                avrgpk_wv=np.array(line)
                continue
            elif len(line)>len(SPIKED_MSTRING) and line[:len(SPIKED_MSTRING)]==SPIKED_MSTRING:
                continue
            else:
                line=line[1:].split(';')
                #print(line)
                for curr in line:
                    curr=curr.split(':')
                    if len(curr)!=2:
                        continue
                    extra_info[curr[0]]=curr[1]
            continue
        # read regular data
        tmp=line.split(';')
        if len(tmp)>=6:
                tmp=float(tmp[4])
                tmp=np.sqrt(tmp)
                soffset.append(tmp)
                if tmp>sigma_cut: # TODO: this removes values that have a sigma on the offset that is bigger than 1
                        continue
        tmp=-float(line.split(';')[2])  # is in mV ns
        tmp=tmp*1e-9/R/e*1e-3           # rewrite to e- count 
        charge.append(tmp)
    f.close()
    return charge,extra_info,avrg_wv,soffset

def do_fit(ffunc,iv,arg):
    """
        wrapper for leastsq that makes the output more pretty
    """
    tmp=leastsq(ffunc,iv,arg,full_output=True)
    if tmp[1] is not None:
        return tmp[0],[np.sqrt(tmp[1][i][i]) for i in range(len(tmp[0]))], \
                sum(tmp[2]['fvec']**2),len(arg[0])-len(iv),np.linspace(min(arg[0]),max(arg[0]),1000)
    return tmp[0],None, \
            sum(tmp[2]['fvec']**2),len(arg[0])-len(iv),np.linspace(min(arg[0]),max(arg[0]),1000)

def val2str(val,err):
    """
        generate a latex string that sanely displays the given value/error pair
    """
    def omag(n):
        if n==0 or n is None or n is np.nan:
            return 0
        return int(np.floor(np.log10(abs(n))))
    omag_val=omag(val)
    if err is not None:
	    omag_err=omag(err)
	    prec=(omag_val-omag_err if omag_val>=omag_err else 0)+1
	    return ('(%.'+str(prec)+'f\\pm%.'+str(prec)+'f)\\cdot 10^{%i}')%(val*10.**(-omag_val),err*10.**(-omag_val),omag_val)
    else:
	    prec=3
	    return ('%.'+str(prec)+'f\\cdot 10^{%i}')%(val*10.**(-omag_val),omag_val)


def find_first(data,trigger,start=0,rev=False):
    """
        returns the index of the first value that is bigger/smaller than trigger
        data      - 1D array containing the data
        trigger   - trigger value
        start     - (Optional) index to start at
    """
    assert start<len(data)
    stp=reversed(range(start,len(data))) if rev else range(start,len(data))
    start=len(data)-1 if rev else start
    if data[start]>trigger:
        for i in stp:
            if data[i]<trigger:
                return i
    else:
        for i in stp:
            if data[i]>trigger:
                return i
    return None

def find_min(data,rnge):
    """
        returns the index of the smalles element of data
    """
    print('rnge',rnge,data[rnge[0]],data[rnge[1]])
    assert rnge[0]>=0 and rnge[1]<len(data)
    cmval=data[rnge[0]]
    cidx=rnge[0]
    print(range(rnge[0],rnge[1]+1))
    for i in range(rnge[0],rnge[1]+1):
        print(data[i],cmval)
        if data[i]<cmval:
            cmval=data[i]
            cidx=int(i)
    return cidx

def find_last_peak(data,start):
    lval=data[start]
    for i in reversed(range(start-1)):
        if data[i]<lval:
            return i
        lval=data[i]
    return None
