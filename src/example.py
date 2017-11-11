#!/user/bin/python3
"""
    minimalistic example
"""
import pmt_sim
import numpy as np
import matplotlib.pyplot as plt

# create dictionary with the simulaiton settings
ss=dict(pmt_sim.DEF_SET)
ss['DYN_FACTORS']=pmt_sim.DYN_FACT_HAM

# run the simulation
data=pmt_sim.pmt_sim(ss,progess_cb=pmt_sim.progress_bar,osf=0.3)
print()

# create histograms
X,per_dyn,full,_=pmt_sim.gen_hist_data(data[1])

# plot histograms
ax=plt.subplot(111)
plt.step(X,full,c='k')
ax.set_yscale('log')
plt.show()
