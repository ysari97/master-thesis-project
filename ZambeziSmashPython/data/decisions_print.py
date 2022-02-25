import numpy as np

data=np.loadtxt('decisions_best_hydro.txt')
index4=0
index5=1
index6a=2
index6b=3
index6c=4
index7a=5
index7b=6
index7c=7
index8=8


#print data[5,:]
array_no_decisions=[196,230,264,298,332]
no_dec_variables=332 # 7 reservoirs
data1= data[index7a,0:298]

np.savetxt('../test_decision_7DM.txt',data1)