import mpathic as mpa
loader = mpa.io

##############################
## load dataframe and model ##
##############################

dataset_df = loader.load_dataset(mpa.__path__[0] + '/data/sortseq/full-0/library.txt')
mp_df = loader.load_model(mpa.__path__[0] + '/examples/true_model.txt')

ss = mpa.SimulateSort(df=dataset_df, mp=mp_df)

print('Library: \n')
print(dataset_df.head(), '\n')

print('Model: \n')
print(mp_df.head(),'\n')

#############################
## simulate sort dataframe ##
#############################

temp_ss = ss.output_df
cols = ['ct', 'ct_0', 'ct_1', 'ct_2', 'ct_3','seq']
temp_ss = temp_ss[cols]

print('Simulate Sort Dataset: \n')
print(temp_ss.head(),'\n')

#################
## learn model ##
#################

lm = mpa.LearnModel(df=temp_ss, lm='LS', alpha=0.001, modeltype='NBR')

print('Learned model: \n')
print(lm.output_df.head(),'\n')

df = lm.output_df
normed_df = (df - df.mean()) / (df.max() - df.min())

import matplotlib.pyplot as plt
import seaborn as sns

heatmap_columns_list = []
for column in normed_df:
    if 'val' in column:
        heatmap_columns_list.append(column)

ax = sns.heatmap(normed_df[heatmap_columns_list],cmap='coolwarm')

####################
## evaluate model ##
####################

em = mpa.EvaluateModel(dataset_df = temp_ss, model_df = mp_df)

print('Evaluated Model: \n')
print(em.out_df.head(),'\n')

################
## scan model ##
################

fastafile = "./mpathic/examples/genome_ecoli_1000lines.fa"
contig = mpa.io.load_contigs_from_fasta(fastafile, mp_df)
sm = mpa.ScanModel(model_df = mp_df, contig_list = contig)

print('Scanned Model: \n')
print(sm.sitelist_df, '\n')

############################
## predictive information ##
############################

pi = mpa.PredictiveInfo(data_df = temp_ss, model_df = mp_df, start=0)

print('Mutual Information: \n')
print(pi.out_MI)

# plot learned model

plt.tight_layout()
plt.show()