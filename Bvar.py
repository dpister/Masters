import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math
import matrixgen as mg

#Font
mg.set_font(legendsize=13,ticksize=12,labelsize=15)

#Variables
N=2                            #number spins         
s=5/2                            #spin
max_val=6                      #number of shown states   
#temp=[1,2,4,8]
temp=[]                 #temperature (for exp-values)
angle=2*math.pi/N
mag=0.6717
b=int(2*s+1) 

#Colors
markers=['.', 'x', 'o', 's', '*', '+', 'p', 'h', 'D']
colors=['red','blue','green','purple','orange','turquoise','magenta','brown','grey']
colors_fixed=['red','blue','green','purple','orange','turquoise','magenta','brown','grey']
col_test=np.zeros(max_val)
col_swap=0.1
col_degen=0.0001
col_switch=True
col_save='placeholder'

#Switches
#exp_switch=range(N)
exp_switch=[]
corr_switch=False
full_switch=False
tor_dir=['z']
mag_dir=[]       
save_switch=True        
   
#Heisenberg interaction matrix
Jconst=1   
Jmatrix=np.zeros([N,N])
for ind in range(N-1):
    Jmatrix[ind+1,ind]=1
Jmatrix[N-1,0]=1
Jmatrix*=Jconst                

#magnetic field
tstart=-3
tfinish=3
tsteps=500
tstepsize=(tfinish-tstart)/tsteps
Bfield_name=r'$\vec{B}^\varphi$'
#Bfield_name=r'$\vec{B}^z$'
#Bfield_var=r'$B^z$ in T'
Bfield_var=r'$B^\varphi$ in T'
Bfield_save='Bp'

Bvec=np.array([1,0,0])
Bvectors=np.zeros([N,3])
Bvectors_circ=np.zeros([N,3])
for ind in range(N):
    vec=np.array([0.00000001,0,0])
    #vec=np.array([0,0,1])
    #vec=np.array([-math.sin(math.pi/N),math.cos(math.pi/N),0])
    rotZ=np.array([[math.cos(ind*angle),-math.sin(ind*angle),0],[math.sin(ind*angle),math.cos(ind*angle),0],[0,0,1]])
    vec_circ=np.dot(rotZ,Bvec)
    Bvectors_circ[ind,:]=vec_circ 
    Bvectors[ind,:]=vec
def Bfield(t): 
    return Bvectors+t*Bvectors_circ

#anisotropy tensor
Dmatrix=np.zeros([N,4])
Dconst=-1
tang=0*2*math.pi/360
Dvec=np.array([math.cos(tang),0,math.sin(tang)])
for ind in range(N):
    Dmatrix[ind,0]=Dconst
    rotZ=np.array([[math.cos(ind*angle),-math.sin(ind*angle),0],[math.sin(ind*angle),math.cos(ind*angle),0],[0,0,1]])
    vec=np.dot(rotZ,Dvec)
    for j in range(3):
        Dmatrix[ind,j+1]=vec[j]
        
#defining r-vectors for toroidal moment
Rmatrix=np.zeros([N,3])
Rvec=np.array([0,1,0])
angle=2*math.pi/N
for ind in range(N):
    #rotX=np.array([[1,0,0],[0,math.cos(ind*angle),-math.sin(ind*angle)],[0,math.sin(ind*angle),math.cos(ind*angle)]])
    #rotY=np.array([[math.cos(ind*angle),0,math.sin(ind*angle)],[0,1,0],[-math.sin(ind*angle),0,math.cos(ind*angle)]])
    rotZ=np.array([[math.cos(ind*angle),-math.sin(ind*angle),0],[math.sin(ind*angle),math.cos(ind*angle),0],[0,0,1]])
    vec=np.dot(rotZ,Rvec)
    Rmatrix[ind,:]=vec
        
#Heisenberg+Anisotropy
m=mg.reverse_trafo(N,s)
hamJD=mg.ham_mat_JD(Jmatrix,1,Dmatrix,N,s,m)
                              
                                                             
#Plots(frame)
####################

fig_eig2, ax_eig2=plt.subplots()
#plt.ylim(-28,-10)
ax_eig2.set_xlabel(Bfield_var)  
ax_eig2.set_ylabel(r'$E_\nu$ in K')  
ax_eig2.set_title(r'Unterste Eigenwerte $E_\nu$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')

if full_switch:
    fig_eig1, ax_eig1=plt.subplots()
    ax_eig1.set_xlabel(Bfield_var)  
    ax_eig1.set_ylabel(r'$E_\nu$ in K')  
    ax_eig1.set_title(r'Eigenwerte $E_\nu$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')

for ind in tor_dir:
    if ind=='x':
        fig_torX, ax_torX=plt.subplots()
        plt.ylim(-2.5,2.5)
        ax_torX.set_xlabel(Bfield_var)
        ax_torX.set_ylabel(r'$<\hat{\tau}^x>_\nu$ in a.u.')
        ax_torX.set_title(r'$<\hat{\tau}^x>_\nu$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
        if len(temp)>0:
            fig_torX_temp, ax_torX_temp=plt.subplots()
            #plt.ylim(-5,5)
            ax_torX_temp.set_xlabel(Bfield_var)
            ax_torX_temp.set_ylabel(r'$<\hat{\tau}^x>(T)$ in a.u.')
            ax_torX_temp.set_title(r'$<\hat{\tau}^x>(T)$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
    if ind=='y':
        fig_torY, ax_torY=plt.subplots()
        plt.ylim(-2.5,2.5)
        ax_torY.set_xlabel(Bfield_var)
        ax_torY.set_ylabel(r'$<\hat{\tau}^y>_\nu$ in a.u.')
        ax_torY.set_title(r'$<\hat{\tau}^y>_\nu$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
        if len(temp)>0:
            fig_torY_temp, ax_torY_temp=plt.subplots()
            #plt.ylim(-5,5)
            ax_torY_temp.set_xlabel(Bfield_var)
            ax_torY_temp.set_ylabel(r'$<\hat{\tau}^y>(T)$ in a.u.')
            ax_torY_temp.set_title(r'$<\hat{\tau}^y>(T)$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
    if ind=='z':
        fig_torZ, ax_torZ=plt.subplots()
        plt.ylim(-5.5,5.5)
        ax_torZ.set_xlabel(Bfield_var)
        ax_torZ.set_ylabel(r'$<\hat{\tau}^z>_\nu$ in a.u.')
        ax_torZ.set_title(r'$<\hat{\tau}^z>_\nu$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
        if len(temp)>0:
            fig_torZ_temp, ax_torZ_temp=plt.subplots()
            #plt.ylim(-5,5)
            ax_torZ_temp.set_xlabel(Bfield_var)
            ax_torZ_temp.set_ylabel(r'$<\hat{\tau}^z>(T)$ in a.u.')
            ax_torZ_temp.set_title(r'$<\hat{\tau}^z>(T)$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
      
ax_expX_array=[]
ax_expY_array=[]
ax_expZ_array=[]
if save_switch:
    fig_expX_array=[]
    fig_expY_array=[]
    fig_expZ_array=[]   
for ind in exp_switch:
    fig_expX, ax_expX=plt.subplots()
    plt.ylim(-3,3)
    ax_expX.set_xlabel(Bfield_var)
    ax_expX.set_ylabel(r'$<\hat{s}^x_%.i>_\nu$'%(ind+1))
    ax_expX.set_title(r'$<\hat{s}^x_%.i>_\nu$'%(ind+1)+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
    ax_expX_array.append(ax_expX)
    if save_switch:
        fig_expX_array.append(fig_expX)
    
    fig_expY, ax_expY=plt.subplots()
    plt.ylim(-3,3)
    ax_expY.set_xlabel(Bfield_var)
    ax_expY.set_ylabel(r'$<\hat{s}^y_%.i>_\nu$'%(ind+1))
    ax_expY.set_title(r'$<\hat{s}^y_%.i>_\nu$'%(ind+1)+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
    ax_expY_array.append(ax_expY)
    if save_switch:
        fig_expY_array.append(fig_expY)
    
    fig_expZ, ax_expZ=plt.subplots()
    plt.ylim(-3,3)
    ax_expZ.set_xlabel(Bfield_var)  
    ax_expZ.set_ylabel(r'$<\hat{s}^z_%.i>_\nu$'%(ind+1)) 
    ax_expZ.set_title(r'$<\hat{s}^z_%.i>_\nu$'%(ind+1)+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
    ax_expZ_array.append(ax_expZ)
    if save_switch:
        fig_expZ_array.append(fig_expZ)
    
if corr_switch:
    ax_corr_array=[] 
    if save_switch:
        fig_corr_array=[]
    for ind in range(N):
        if ind==N-1:
            if N!=2:
                fig_corr, ax_corr=plt.subplots()
                #plt.ylim(-2,2)
                ax_corr.set_xlabel(Bfield_var)  
                ax_corr.set_ylabel(r'$<\hat{\vec{s}}_1\cdot\hat{\vec{s}}_%.i>_\nu$'%(N))
                ax_corr.set_title(r'$<\hat{\vec{s}}_1\cdot\hat{\vec{s}}_%.i>_\nu$'%(N)+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
                ax_corr_array.append(ax_corr)
                if save_switch:
                    fig_corr_array.append(fig_corr)
        else:
            fig_corr, ax_corr=plt.subplots()
            #plt.ylim(-2,2)
            ax_corr.set_xlabel(Bfield_var)
            ax_corr.set_ylabel(r'$<\hat{\vec{s}}_%.i\cdot\hat{\vec{s}}_%.i>_\nu$'%(ind+1,ind+2))
            ax_corr.set_title(r'$<\hat{\vec{s}}_%.i\cdot\hat{\vec{s}}_%.i>_\nu$'%(ind+1,ind+2)+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
            ax_corr_array.append(ax_corr)
            if save_switch:
                    fig_corr_array.append(fig_corr)
         
for ind in mag_dir:
    if ind=='x' and len(temp)>0:
         fig_magX_temp, ax_magX_temp=plt.subplots()
         plt.ylim(-2,2)
         ax_magX_temp.set_xlabel(Bfield_var)
         ax_magX_temp.set_ylabel(r'$<\mathcal{M}^x>(T)$ in J/K')
         ax_magX_temp.set_title(r'$<\mathcal{M}^x>(T)$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
    if ind=='y' and len(temp)>0:
         fig_magY_temp, ax_magY_temp=plt.subplots()
         plt.ylim(-2,2)
         ax_magY_temp.set_xlabel(Bfield_var)
         ax_magY_temp.set_ylabel(r'$<\mathcal{M}^y>(T)$ in J/K')
         ax_magY_temp.set_title(r'$<\mathcal{M}^y>(T)$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')
    if ind=='z' and len(temp)>0:
         fig_magZ_temp, ax_magZ_temp=plt.subplots()
         plt.ylim(-2,2)
         ax_magZ_temp.set_xlabel(Bfield_var)
         ax_magZ_temp.set_ylabel(r'$<\mathcal{M}^z>(T)$ in J/K')
         ax_magZ_temp.set_title(r'$<\mathcal{M}^z>(T)$'+f' für N={N}, s={s}, J={Jconst} K, D={Dconst} K, '+r'$\vec{B}=$'+f'{Bfield_name}')


#Zeeman
t_array=np.linspace(-tstepsize/2,tstart,int(tsteps/2))
t_array=np.append(t_array,0.000001)
ls=np.linspace(tstepsize/2,tfinish,int(tsteps/2))
t_array=np.append(t_array,ls)

for t in t_array:
    hamB=mg.ham_mat_B(Bfield,t,N,s,m)

    #Eigenvalues
    hamJDB=hamB+hamJD
    vals,vectors=la.eig(hamJDB)
    vals=np.real(vals)
    order=np.argsort(vals)     #place, in which the n-lowest number is 
    vals=np.sort(vals)
    if t==0.000001:
        print(vals[1]-vals[0])
        
    #color coding
    if col_switch:
        if t==tstart:
            for ind1 in range(max_val-1):
                if abs(vals[ind1]-vals[ind1+1])<col_degen: #degenerate states
                    col_test[ind1]=2
        for ind1 in range(max_val-1):
            if abs(vals[ind1]-vals[ind1+1])<col_degen: #degenerate states
                col_test[ind1]=2
            if col_degen<abs(vals[ind1]-vals[ind1+1])<col_swap and col_test[ind1]==0: #level crossing
                col_test[ind1]=1 #locking
                index=ind1
                ind2=ind1
                while col_test[index+1]!=0: #counting upper level degeneracy
                    index+=1
                while col_test[ind2-1]!=0: #counting lower level degeneracy
                    ind2-=1
                for j in range(index-ind1+1): #putting upper level under lower level
                    bruh=colors.pop(ind1+1+j)
                    colors.insert(ind2+j,bruh)
            '''if col_degen<abs(vals[ind1]-vals[ind1+1])<col_swap and col_test[ind1]==2: #level touching (NOT WORKING,just an attempt)
                col_test[ind1]=1 #locking
                index=ind1
                ind2=ind1
                while col_test[index+1]!=0: #counting upper level degeneracy
                    index+=1
                while col_test[ind2-1]!=0: #counting lower level degeneracy
                    ind2-=1
                for j in range(index-ind1+1): #putting upper level under lower level
                    bruh=colors.pop(ind1+1+j)
                    colors.insert(ind2+j,bruh)'''         
            if abs(vals[ind1]-vals[ind1+1])>col_swap and col_test[ind1]!=0: #unlocking
                col_test[ind1]=0 
                
        if t==0.000001:
                colors=[]
                colors=colors+col_save
        if abs(t)<tstepsize:
            col_save=[]
            col_save=col_save+colors

    Z=np.zeros(len(temp))
    j=0
    for ind1 in temp:
        for ind2 in range(b**N):
            Z[j]+=math.exp(-1/ind1*vals[ind2])
        j+=1
    if full_switch:
        tarr1=t*np.ones(b**N)
        ax_eig1.scatter(tarr1, vals, color='blue', marker='.', linewidths=0.00001)
    if col_switch:
        for ind1 in range(max_val):
            ax_eig2.scatter(t,vals[ind1] , color=colors[ind1], marker='.', linewidths=0.00001)
    else:
        tarr2=t*np.ones(max_val)
        ax_eig2.scatter(tarr2,vals[:max_val] , color='blue', marker='.', linewidths=0.00001)
    
    #toroidal moment
    for ind1 in range(max_val):
        tor_res=np.real(mg.tor_exp(ind1,Rmatrix,N,s,m,vectors,order))
        for ind2 in tor_dir:
            if ind2=='x':
                ax_torX.scatter(t,tor_res[0], color=colors[ind1], marker='.', linewidths=0.000001)
            if ind2=='y':
                ax_torY.scatter(t,tor_res[1], color=colors[ind1], marker='.', linewidths=0.000001)
            if ind2=='z':
                ax_torZ.scatter(t,tor_res[2], color=colors[ind1], marker='.', linewidths=0.000001)  
    j=0
    for ind1 in temp:
        tor_temp_res=np.zeros(3)
        for ind2 in range(b**N):
            tor_exp_res=np.real(mg.tor_exp(ind2,Rmatrix,N,s,m,vectors,order))
            tor_temp_res[0]+=math.exp(-vals[ind2]/ind1)*tor_exp_res[0]
            tor_temp_res[1]+=math.exp(-vals[ind2]/ind1)*tor_exp_res[1]
            tor_temp_res[2]+=math.exp(-vals[ind2]/ind1)*tor_exp_res[2]
        tor_temp_res/=Z[j]
        for ind2 in tor_dir:
            if ind2=='x':
                ax_torX_temp.scatter(t,tor_temp_res[0], color=colors_fixed[j], marker='.', linewidths=0.000001)
            if ind2=='y':
                ax_torY_temp.scatter(t,tor_temp_res[1], color=colors_fixed[j], marker='.', linewidths=0.000001)
            if ind2=='z':
                ax_torZ_temp.scatter(t,tor_temp_res[2], color=colors_fixed[j], marker='.', linewidths=0.000001)
        j+=1
              
    #spin expectation values
    for ind1 in exp_switch:
        for ind2 in range(max_val):
            exp_res=np.real(mg.spin_exp(ind1,ind2,N,s,m,vectors,order))     
            ax_expX_array[ind1].scatter(t, exp_res[0], color=colors[ind2] , marker='.', linewidths=0.000001)
            ax_expY_array[ind1].scatter(t, exp_res[1], color=colors[ind2] , marker='.', linewidths=0.000001)
            ax_expZ_array[ind1].scatter(t, exp_res[2], color=colors[ind2] , marker='.', linewidths=0.000001)
    
    #correlation functions
    if corr_switch:
        for ind1 in range(N):
            if ind1==N-1:
                if N!=2:
                    for ind2 in range(max_val):
                        ax_corr_array[ind1].scatter(t, mg.corr_func(0,N-1,ind2,N,s,m,vectors,order).real, color=colors[ind2] , marker='.', linewidths=0.000001)
            else:
                for ind2 in range(max_val):
                    ax_corr_array[ind1].scatter(t, mg.corr_func(ind1,ind1+1,ind2,N,s,m,vectors,order).real, color=colors[ind2] , marker='.', linewidths=0.000001)
    
    #magnetization
    j=0
    for ind1 in temp:
        mag_temp_res=np.zeros(3)
        for ind2 in range(N):
            for ind3 in range(b**N):
                spin_exp_res=np.real(mg.spin_exp(ind2,ind3,N,s,m,vectors,order))
                mag_temp_res[0]+=math.exp(-vals[ind3]/ind1)*spin_exp_res[0]
                mag_temp_res[1]+=math.exp(-vals[ind3]/ind1)*spin_exp_res[1]
                mag_temp_res[2]+=math.exp(-vals[ind3]/ind1)*spin_exp_res[2]
        mag_temp_res*=(-2*mag/Z[j])
        for ind2 in mag_dir:
             if ind2=='x':
                ax_magX_temp.scatter(t,mag_temp_res[0], color=colors_fixed[j], marker='.', linewidths=0.000001)
             if ind2=='y':
                ax_magY_temp.scatter(t,mag_temp_res[1], color=colors_fixed[j], marker='.', linewidths=0.000001)
             if ind2=='z':
                ax_magZ_temp.scatter(t,mag_temp_res[2], color=colors_fixed[j], marker='.', linewidths=0.000001)
        j+=1
               

#Labels
#######        
    
labels_nu=[]
for ind in range(max_val):
    labels_nu.append(r'$\nu=$'+str(ind))
    
labels_temp=[]
for ind in temp:
    labels_temp.append('$T=$'+str(ind)+' K')
    
if col_switch:
    ax_eig2.legend(labels_nu,loc='upper right')
    
for ind in exp_switch:    
    ax_expX_array[ind].legend(labels_nu,loc='upper right')
    ax_expY_array[ind].legend(labels_nu,loc='upper right')
    ax_expZ_array[ind].legend(labels_nu,loc='upper right')
    
if corr_switch:
    for ind in range(N):
        if ind==N-1:
            if N!=2:
                ax_corr_array[ind].legend(labels_nu,loc='upper right')
        else:
            ax_corr_array[ind].legend(labels_nu,loc='upper right') 
    
        
for ind in tor_dir:
    if ind=='x':
        ax_torX.legend(labels_nu,loc='upper right')
        if len(temp)>0:
            ax_torX_temp.legend(labels_temp,loc='upper right')
    if ind=='y':
        ax_torY.legend(labels_nu,loc='upper right')
        if len(temp)>0:
            ax_torY_temp.legend(labels_temp,loc='upper right')
    if ind=='z':
        ax_torZ.legend(labels_nu,loc='upper right')
        if len(temp)>0:
            ax_torZ_temp.legend(labels_temp,loc='upper right')
            
for ind in mag_dir:
    if ind=='x' and len(temp)>0:
        ax_magX_temp.legend(labels_temp,loc='upper right')
    if ind=='y' and len(temp)>0:
        ax_magY_temp.legend(labels_temp,loc='upper right')
    if ind=='z' and len(temp)>0:
        ax_magZ_temp.legend(labels_temp,loc='upper right')


#Save
#####

if save_switch:
    fig_eig2.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_elow.png',bbox_inches='tight')
    if full_switch:
        fig_eig1.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_evals.png',bbox_inches='tight')
    
    for ind in tor_dir:
        if ind=='x':
            fig_torX.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_Tx.png',bbox_inches='tight')
            if len(temp)>0:
                fig_torX_temp.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_TxT.png',bbox_inches='tight')
        if ind=='y':
            fig_torY.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_Ty.png',bbox_inches='tight')
            if len(temp)>0:
                fig_torY_temp.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_TyT.png',bbox_inches='tight')
        if ind=='z':
            fig_torZ.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_Tz.png',bbox_inches='tight')
            if len(temp)>0:
                fig_torZ_temp.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_TzT.png',bbox_inches='tight')

    for ind in exp_switch:
        fig_expX_array[ind].savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+f'_{ind+1}x.png',bbox_inches='tight')
        fig_expY_array[ind].savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+f'_{ind+1}y.png',bbox_inches='tight')
        fig_expZ_array[ind].savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+f'_{ind+1}z.png',bbox_inches='tight')
        
    for ind in mag_dir:
        if ind=='x' and len(temp)>0:
            fig_magX_temp.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_Mx.png',bbox_inches='tight')
        if ind=='y' and len(temp)>0:
            fig_magY_temp.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_My.png',bbox_inches='tight')
        if ind=='z' and len(temp)>0:
            fig_magZ_temp.savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+'_Mz.png',bbox_inches='tight')
            
    if corr_switch:
        for ind in range(N):
            if ind==N-1:
                if N!=2:
                    fig_corr_array[ind].savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+f'_1{N}.png',bbox_inches='tight')
            else:
                fig_corr_array[ind].savefig(f's{s}_J{Jconst}_D{Dconst}_'+Bfield_save+f'_{ind+1}{ind+2}.png',bbox_inches='tight')
    mg.ping()
        
else:                
    plt.show()
