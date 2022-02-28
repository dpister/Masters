import numpy as np
import math
import matplotlib.pyplot as plt
from win10toast import ToastNotifier as tn


def set_font(legendsize,ticksize,labelsize):
    plt.rc('axes', titlesize=labelsize)     # fontsize of the axes title
    plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=ticksize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=ticksize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=legendsize)    # legend fontsize
    return

#reverse transformation
def reverse_trafo(N,s):
    
    b=int(2*s+1)
    m=-s*np.ones([N,b**N])
    
    for x in range(b**N):
        xsave=x
        ind=0                            
        while x:
            m[ind,xsave]=int(x % b)-s
            x //= b
            ind+=1
    return m


#spin expectation value
def spin_exp(spin_ind,val_ind,N,s,m,vectors,order):
    
    b=int(2*s+1)
    spin_vec=np.zeros(3)+np.zeros(3)*1j
    
    for c in range(b**N):
        for d in range(b**N):
            for ind in range(N):
                if ind==spin_ind:
                    if m[ind,c]!=m[ind,d]+1:
                        break
                else:
                    if m[ind,c]!=m[ind,d]:
                        break
                if ind==N-1:
                    spin_vec[0]+=1/2*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]*math.sqrt(s*(s+1)-m[spin_ind,d]*(m[spin_ind,d]+1))
                    spin_vec[1]+=-1j/2*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]*math.sqrt(s*(s+1)-m[spin_ind,d]*(m[spin_ind,d]+1))
            for ind in range(N):
                if ind==spin_ind:
                    if m[ind,c]!=m[ind,d]-1:
                        break
                else:
                    if m[ind,c]!=m[ind,d]:
                        break
                if ind==N-1:
                    spin_vec[0]+=1/2*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]*math.sqrt(s*(s+1)-m[spin_ind,d]*(m[spin_ind,d]-1))
                    spin_vec[1]+=1j/2*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]*math.sqrt(s*(s+1)-m[spin_ind,d]*(m[spin_ind,d]-1))
            if c==d:
                spin_vec[2]+=m[spin_ind,c]*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]
    return spin_vec
               

#correlation function
def corr_func(spin_ind1,spin_ind2,val_ind,N,s,m,vectors,order):
    
    b=int(2*s+1)
    corr_value=0
    
    if spin_ind1==spin_ind2:
        return 'choose 2 different spins'    
    for c in range(b**N):
        for d in range(b**N):
            for ind in range(N):
                if ind==spin_ind1:
                    if m[ind,c]!=m[ind,d]+1:
                        break
                elif ind==spin_ind2:
                    if m[ind,c]!=m[ind,d]-1:
                        break                       
                else:
                    if m[ind,c]!=m[ind,d]:
                        break
                if ind==N-1:
                    corr_value+=1/2*math.sqrt(s*(s+1)-m[spin_ind1,d]*(m[spin_ind1,d]+1))*math.sqrt(s*(s+1)-m[spin_ind2,d]*(m[spin_ind2,d]-1))*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]
            for ind in range(N):
                if ind==spin_ind1:
                    if m[ind,c]!=m[ind,d]-1:
                        break
                elif ind==spin_ind2:        
                    if m[ind,c]!=m[ind,d]+1:
                        break
                else:
                    if m[ind,c]!=m[ind,d]:
                        break
                if ind==N-1:
                    corr_value+=1/2*math.sqrt(s*(s+1)-m[spin_ind1,d]*(m[spin_ind1,d]-1))*math.sqrt(s*(s+1)-m[spin_ind2,d]*(m[spin_ind2,d]+1))*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]
            if c==d:
                corr_value+=m[spin_ind1,c]*m[spin_ind2,c]*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]
    return corr_value

        
#toroidal moment expectation value
def tor_exp(val_ind,Rmatrix,N,s,m,vectors,order):
    
    b=int(2*s+1)
    tor_vec=np.zeros(3)+np.zeros(3)*1j
    
    for spin_ind in range(N):
        for c in range(b**N):
            for d in range(b**N):
                for ind in range(N):
                    if ind==spin_ind:
                        if m[ind,c]!=m[ind,d]+1:
                            break
                    else:
                        if m[ind,c]!=m[ind,d]:
                            break
                    if ind==N-1:
                        tor_vec[0]+=1j/2*Rmatrix[spin_ind,2]*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]*math.sqrt(s*(s+1)-m[spin_ind,d]*(m[spin_ind,d]+1))
                        tor_vec[1]+=Rmatrix[spin_ind,2]*1/2*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]*math.sqrt(s*(s+1)-m[spin_ind,d]*(m[spin_ind,d]+1))
                        tor_vec[2]+=(-1j/2*Rmatrix[spin_ind,0]-1/2*Rmatrix[spin_ind,1])*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]*math.sqrt(s*(s+1)-m[spin_ind,d]*(m[spin_ind,d]+1))
                for ind in range(N):
                    if ind==spin_ind:
                        if m[ind,c]!=m[ind,d]-1:
                            break
                    else:
                        if m[ind,c]!=m[ind,d]:
                            break
                    if ind==N-1:
                        tor_vec[0]+=-1j/2*Rmatrix[spin_ind,2]*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]*math.sqrt(s*(s+1)-m[spin_ind,d]*(m[spin_ind,d]-1))
                        tor_vec[1]+=1/2*Rmatrix[spin_ind,2]*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]*math.sqrt(s*(s+1)-m[spin_ind,d]*(m[spin_ind,d]-1))
                        tor_vec[2]+=(1j/2*Rmatrix[spin_ind,0]-1/2*Rmatrix[spin_ind,1])*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]*math.sqrt(s*(s+1)-m[spin_ind,d]*(m[spin_ind,d]+1))
                if c==d:
                    tor_vec[0]+=Rmatrix[spin_ind,1]*m[spin_ind,c]*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]
                    tor_vec[1]+=-Rmatrix[spin_ind,0]*m[spin_ind,c]*vectors[c,order[val_ind]].conjugate()*vectors[d,order[val_ind]]
    return tor_vec


#Matrix with J-Term and D-Term (XXZ-Model)
def ham_mat_JD(Jmatrix,fac,Dmatrix,N,s,m):
    
    b=int(2*s+1)
    hamJD=np.zeros([b**N,b**N])+np.zeros([b**N,b**N])*1j

    for c in range(b**N):
        for d in range(b**N):
        
            #diagonal terms
            if c==d: 
                for ind1 in range(N):
                    ind2=0
                    while ind2<ind1:
                        hamJD[c,d]+=m[ind1,d]*m[ind2,d]*Jmatrix[ind1,ind2]*fac 
                        ind2+=1
                    hamJD[c,d]+=Dmatrix[ind1,0]*Dmatrix[ind1,3]**2*m[ind1,d]**2
                    hamJD[c,d]+=Dmatrix[ind1,0]*(2*s*(s+1)-2*m[ind1,d]**2)*(1/4*Dmatrix[ind1,1]**2+1/4*Dmatrix[ind1,2]**2) 
            
            #Heisenberg rest 
            for ind1 in range(N):
                ind2=0
                while ind2<ind1:
                    for ind3 in range(N):
                        if ind3==ind1:
                            if m[ind3,c]!=m[ind3,d]+1:
                                break
                        elif ind3==ind2:
                            if m[ind3,c]!=m[ind3,d]-1:
                                break                       
                        else:
                            if m[ind3,c]!=m[ind3,d]:
                                break
                        if ind3==N-1:
                            hamJD[c,d]+=Jmatrix[ind1,ind2]/2*math.sqrt(s*(s+1)-m[ind1,d]*(m[ind1,d]+1))*math.sqrt(s*(s+1)-m[ind2,d]*(m[ind2,d]-1))
                    for ind3 in range(N):
                        if ind3==ind1:
                            if m[ind3,c]!=m[ind3,d]-1:
                                break
                        elif ind3==ind2:        
                            if m[ind3,c]!=m[ind3,d]+1:
                                break
                        else:
                            if m[ind3,c]!=m[ind3,d]:
                                break
                        if ind3==N-1:
                            hamJD[c,d]+=Jmatrix[ind1,ind2]/2*math.sqrt(s*(s+1)-m[ind1,d]*(m[ind1,d]-1))*math.sqrt(s*(s+1)-m[ind2,d]*(m[ind2,d]+1))
                    ind2+=1
                    
            #Anisotropy rest
            for ind1 in range(N):
                for ind2 in range(N):
                    if ind2==ind1:
                        if m[ind2,c]!=m[ind2,d]+1:
                            break
                    else:
                        if m[ind2,c]!=m[ind2,d]:
                            break
                    if ind2==N-1:
                        hamJD[c,d]+=Dmatrix[ind1,0]*math.sqrt(s*(s+1)-m[ind1,d]*(m[ind1,d]+1))*(2*m[ind1,d]+1)*(1/2*Dmatrix[ind1,1]*Dmatrix[ind1,3]-1j/2*Dmatrix[ind1,2]*Dmatrix[ind1,3])
                for ind2 in range(N):
                    if ind2==ind1:
                        if m[ind2,c]!=m[ind2,d]-1:
                            break
                    else:
                        if m[ind2,c]!=m[ind2,d]:
                            break
                    if ind2==N-1:
                        hamJD[c,d]+=Dmatrix[ind1,0]*math.sqrt(s*(s+1)-m[ind1,d]*(m[ind1,d]-1))*(2*m[ind1,d]-1)*(1/2*Dmatrix[ind1,1]*Dmatrix[ind1,3]+1j/2*Dmatrix[ind1,2]*Dmatrix[ind1,3])        
                for ind2 in range(N):
                    if ind2==ind1:
                        if m[ind2,c]!=m[ind2,d]+2:
                            break
                    else:
                        if m[ind2,c]!=m[ind2,d]:
                            break
                    if ind2==N-1:
                        hamJD[c,d]+=Dmatrix[ind1,0]*math.sqrt(s*(s+1)-m[ind1,d]*(m[ind1,d]+1))*math.sqrt(s*(s+1)-(m[ind1,d]+1)*(m[ind1,d]+2))*(1/4*Dmatrix[ind1,1]**2-1/4*Dmatrix[ind1,2]**2-1j/2*Dmatrix[ind1,1]*Dmatrix[ind1,2])
                for ind2 in range(N):
                    if ind2==ind1:
                        if m[ind2,c]!=m[ind2,d]-2:
                            break
                    else:
                        if m[ind2,c]!=m[ind2,d]:
                            break
                    if ind2==N-1:            
                        hamJD[c,d]+=Dmatrix[ind1,0]*math.sqrt(s*(s+1)-m[ind1,d]*(m[ind1,d]-1))*math.sqrt(s*(s+1)-(m[ind1,d]-1)*(m[ind1,d]-2))*(1/4*Dmatrix[ind1,1]**2-1/4*Dmatrix[ind1,2]**2+1j/2*Dmatrix[ind1,1]*Dmatrix[ind1,2])
                        
    return hamJD

def ham_mat_B(Bfield,t,N,s,m):
    
    b=int(2*s+1)
    hamB=np.zeros([b**N,b**N])+np.zeros([b**N,b**N])*1j

    for c in range(b**N):
        for d in range(b**N):
            for ind1 in range(N):
                if c==d:
                    hamB[c,d]+=2*0.6717*m[ind1,d]*Bfield(t)[ind1,2]
                for ind2 in range(N):
                    if ind2==ind1:
                        if m[ind2,c]!=m[ind2,d]+1:
                            break
                    else:
                        if m[ind2,c]!=m[ind2,d]:
                            break
                    if ind2==N-1:
                        hamB[c,d]+=math.sqrt(s*(s+1)-m[ind1,d]*(m[ind1,d]+1))*0.6717*(Bfield(t)[ind1,0]-Bfield(t)[ind1,1]*1j)
                for ind2 in range(N):
                    if ind2==ind1:
                        if m[ind2,c]!=m[ind2,d]-1:
                            break
                    else:
                        if m[ind2,c]!=m[ind2,d]:
                            break
                    if ind2==N-1:
                        hamB[c,d]+=math.sqrt(s*(s+1)-m[ind1,d]*(m[ind1,d]-1))*0.6717*(Bfield(t)[ind1,0]+Bfield(t)[ind1,1]*1j)
    
    return hamB

def ping():
    toast = tn()
    toast.show_toast("Program finished running",
                     "I have finished your tasks.",
                     duration=5)