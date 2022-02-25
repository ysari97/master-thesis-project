from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import os
#import pandas as pd

sns.set_style()

def parallel_plots(objectives_df):
    file_name='Best_objectives'

    names=['Hydropower','Environment','Irrigation']
    units=['TWh/year','Deficit (cm/sec)'+r'$^2$','Normalized Deficit']

    mx=[]
    mn=[]
    for column in names:
        mx.append(str(round(objectives_df[column].max(), 1)))
        mn.append(str(round(objectives_df[column].min(), 1)))

    objectives_df=(objectives_df.max()-objectives_df)/(objectives_df.max()-objectives_df.min())
    objectives_df['Name'] = "All Solutions"
    for column in names:
        objectives_df = objectives_df.append(objectives_df.loc[objectives_df[column] == 1,:], ignore_index=True)
        objectives_df.iloc[-1,-1] = "Best " + column

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    gray='#bdbdbd'
    purple='#7a0177'
    green='#41ab5d'
    blue='#1d91c0'
    yellow='#fdaa09'
    pink='#c51b7d'

    parallel_coordinates(objectives_df,'Name', color= [gray,purple,yellow,blue], linewidth=7, alpha=.8)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=1.5, fontsize=18)
    
    i=0
    ax1.set_xticks(np.arange(3))


    ax1.set_xticklabels([mx[i]+'\n'+'\n'+names[i]+'\n'+units[i], mx[i+1]+'\n'+'\n'+names[i+1]+'\n'+units[i+1],mx[i+2]+'\n'+'\n'+names[i+2]+'\n'+units[i+2]], fontsize=18)
    ax2 = ax1.twiny()
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels([mn[i], mn[i+1],mn[i+2]], fontsize=18)
    ax1.get_yaxis().set_visible([])
    plt.text(1.02, 0.5, 'Direction of Preference $\\rightarrow$', {'color': '#636363', 'fontsize': 20},
             horizontalalignment='left',
             verticalalignment='center',
             rotation=90,
             clip_on=False,
             transform=plt.gca().transAxes)

    fig.set_size_inches(17.5, 9)


def plot_quantities():

    plt.rcParams["font.family"] = "Myriad Pro"
    sns.set_style("whitegrid")

    input_folder= '../storage_release/'
    #input_folder_objs='../for_plots/'
    target_input_folder='../data/'
    output_folder='../plots/'
    delta_target=np.loadtxt(target_input_folder+'MEF_delta.txt')
    #n_objs=3

    #copy here..
    #####################################
    feature='three_policy_simulation'
    #reservoirs=['itt','kgu','kgl','ka','bg','dg','cb','mn']
    reservoirs=['itt','kgu','kgl','ka','cb']
    title='5_res_wKGL'
    #input_file='Zambezi_'+title+'.reference'  #'.reference'change filename

    # data= np.loadtxt('../parallel/sets/'+feature+'/'+input_file, skiprows=0+1+2-1)
    delta_release_balance='\n('r'$r_{CB}+Q_{Shire}-r_{Irrd7}-r_{Irrd8}-r_{Irrd9}$)'
    #res_names=['Itezhitezhi','Kafue G. Upper','Kafue G. Lower','Kariba','Batoka Gorge','Devil\'s Gorge','Cahora Bassa', 'Mphanda Nkuwa']
    res_names=['Itezhitezhi','Kafue G. Upper','Kafue G. Lower','Kariba','Cahora Bassa']
    #####copy the segment above#########

    policies=['best_hydro', 'best_env', 'best_irr']
    irr_index=['2','3','4','5','6','7','8','9']
    irr_d=['Irrigation District 2', 'Irrigation District 3', 'Irrigation District 4', 'Irrigation District 5','Irrigation District 6','Irrigation District 7','Irrigation District 8', 'Irrigation District 9']
    label_policy=['Best Hydropower', 'Best Environment', 'Best Irrigation', 'Target Demand']
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    n_months=12
    n_years=20
    purple='#7a0177';yellow='#fdaa09';blue='#1d91c0' #green='#41ab5d'
    colors=[purple,yellow,blue]
    variables_names=[r'$q_t$',r'$h_t$',r'$s_t$',r'$s_{t+1}$',r'$r_{t+1}$',r'$r^{delay}_{t+1}$']
    variables=['q','h_t','s_t','s_t+1','r_t+1','r_d_t+1']


    image_format=['png']
    for im in range(len(image_format)):
        for policy in range(len(policies)):
            if not os.path.exists(output_folder+'/'+feature+'/'+image_format[im]+'/'+policies[policy]):
                    os.makedirs(output_folder+'/'+feature+'/'+image_format[im]+'/'+policies[policy])
    #this generates 8 plots one for each irrigation district:
    for ir in range(len(irr_d)):
        irr_plots(input_folder,target_input_folder,output_folder,feature,ir,irr_d,irr_index,policies,months,label_policy,n_months,n_years,colors)

    # this a summary of the delta releases:
    mef_plots(input_folder,output_folder,label_policy,delta_release_balance,feature, policies, n_years,n_months,delta_target,colors, months, title,target_input_folder)

    v=4 # to print only releases across all reservoirs:
    #for v in range(len(variables)-1): to print all summary figures:
    for p in range(len(policies)):
        fig= plt.figure()
        for r in range(len(reservoirs)):
            summary_plot(v,p,r,fig,input_folder,output_folder,feature,policies,variables,label_policy,reservoirs, res_names,months,n_years,n_months)

def irr_plots(input_folder,t_irr_folder,output_folder,feature,ir,irr_d,irr_index,policies,months,label_policy,n_months,n_years, colors):
	left=0.05; bottom=0.17; right=0.98; top=0.89; wspace=0.2; hspace=0.2
	font_size=22
	font_sizey=22
	font_size_title=25

	#for ir in range(len(irr_d)):
	fig = plt.figure()
	for p in range(len(policies)):
	#actual release for irrigation:	
		data=np.loadtxt(input_folder+feature+'/irr_'+policies[p]+'.txt')
	#irrigation target demand:
		data2=np.loadtxt(t_irr_folder+'IrrDemand'+irr_index[ir]+'.txt')
		irrigation=np.reshape(data[:,ir],(n_years,n_months))

		mean_irr=np.mean(irrigation,0)
		min_irr=np.min(irrigation,0)
		max_irr=np.max(irrigation,0)

		plt.fill_between(range(n_months),max_irr,min_irr, alpha=0.5,color=colors[p])
		plt.plot(mean_irr, linewidth=5,color=colors[p], label=label_policy[p])
	
		plt.title(irr_d[ir], fontsize=font_size_title)
		plt.ylabel('Average diversion bounded \nby min and max values [m'r'$^3$/sec]', fontsize=font_sizey, labelpad=20)
		plt.xticks(np.arange(n_months),months, rotation=30, fontsize=font_size)
		plt.yticks(fontsize=font_size)
		plt.xlim([0,11])
	
	plt.plot(data2, color='k', linestyle=':', linewidth=5, label='Target Demand')
	plt.legend(fontsize=font_size)
	fig.set_size_inches(12,10)
	return plt.savefig('../plots/'+feature+'/png/irr_d_'+irr_index[ir]+'.png')

def mef_plots(input_folder,output_folder,label_policy,delta_release_balance,feature, policies, n_years,n_months,delta_target,colors, months, title, mef_folder):
	left=0.18; bottom=0.1; right=0.96; top=0.92; wspace=0.2; hspace=0.2
	font_size=22
	font_sizey=22
	font_size_title=25
	fig = plt.figure()
	for p in range(len(policies)):
		#actual release for mef_
		data=np.loadtxt(input_folder+feature+'/rDelta_'+policies[p]+'.txt')
		#target mef
		rMEF=np.reshape(data,(n_years,n_months))
		mean_mef=np.mean(rMEF,0)
		min_mef=np.min(rMEF,0)
		max_mef=np.max(rMEF,0)

		plt.fill_between(range(n_months),max_mef,min_mef, alpha=0.5,color=colors[p])
		plt.plot(mean_mef, linewidth=5,color=colors[p], label=label_policy[p])
		
		plt.title('Delta releases-'+title+delta_release_balance, fontsize=font_size_title)
		plt.ylabel('Average environmental flows bounded\nby min and max values [m'r'$^3$/sec]', fontsize=font_sizey, labelpad=20)
		plt.xticks(np.arange(n_months),months, rotation=30, fontsize=font_size)
		plt.yticks(fontsize=font_size)
		plt.xlim([0,11])
		
	plt.plot(delta_target, color='k', linestyle=':', linewidth=6, label='MEF Delta target')
	plt.legend(fontsize=font_size)
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	fig.set_size_inches(12,10)
	return plt.savefig(output_folder+feature+'/png/rMEF.png')

def summary_plot(v,p,r,fig,input_folder,output_folder,feature,policies,variables,label_policy,reservoirs, res_names, months,n_years,n_months):
	colorsr=['#b2182b','#d6604d','#fc8d59','#f4a582','#92c5de','#6baed6','#4393c3','#2166ac']
	left=0.13; bottom=0.12; right=0.75; top=0.95; wspace=0.2; hspace=0.2
	font_size=18
	font_sizey=20
	font_size_title=25
	y_label=['Inflow [m'r'$^3$/sec]','Level (t) [m]','Storage (t) [m'r'$^3$]','Storage (t+1) [m'r'$^3$]','Average Release (t+1) [m'r'$^3$/sec]', 'Average Release (t+2) [m'r'$^3$/sec]']
	locs, labels = plt.xticks()
	data=np.loadtxt(input_folder+feature+'/'+reservoirs[r]+'_'+policies[p]+'.txt')
	data=np.reshape(data[:,v],(n_years,n_months))
	avg=np.mean(data,0)
	plt.plot(avg,color=colorsr[r],linewidth=7,linestyle=':',label=res_names[r])
	plt.xticks(np.arange(n_months),months, rotation=30, fontsize=font_size)
	plt.ylabel(y_label[v], fontsize=font_size_title,labelpad=30)
	plt.yticks(fontsize=font_sizey)
	plt.title(label_policy[p],fontsize=font_size_title)
	plt.xlim([0,11])
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	fig.set_size_inches(14,10)
	plt.legend(fontsize=font_sizey,labelspacing=3, loc=6, bbox_to_anchor=(1, 0.5)) #bbox_to_anchor=(0., 1.02, 1., .102), loc=3,fontsize=font_sizey, ncol=4, mode="expand")

	return plt.savefig(output_folder+feature+'/png/'+variables[v]+'_all_reservoirs_'+policies[p]+'.png')


