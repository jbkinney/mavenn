import numpy as np
import fileinput
import pickle


def filter_mutations(x):
    assert isinstance(x, str)
    return x[17:47]

def make_Sequence_Hash(file_name):
    cdr1_start = 'GTGT'
    cdr3_start = 'GTGA'
    sequences = ["ACCCCAGTAGTCCATACCATAGTAAGAACC", "ACTTTTAGTGACTACTGGATGAACTGGGTC"]
    sequence_hash = {sequences[0]: 0, sequences[1]: 1}
    cdr = [3, 1]
    count = len(sequences)

    for line in fileinput.input([file_name]):
        mutation = filter_mutations(line)
        if mutation not in sequence_hash:
            sequence_hash[mutation] = count
            sequences.append(mutation)
            start = line[0:4]
            if start == cdr1_start:
                cdr.append(1)
            elif start == cdr3_start:
                cdr.append(3)

            count += 1

    fileinput.close()
    return sequence_hash, sequences, cdr

nt_complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', '-': '-'}

def comp_rev(x):
    x=x[::-1]
    x=[nt_complement[ii] for ii in x]
    x=''.join(x)
    return x
    
def check_nt2aa(codon):
    if 'N' in codon:
        return 'X'
    elif '-' in codon:
        return '-'
    elif len(codon)!=3:
        return ''
    else:
        return nt_aa_dict[codon]


def nt2aa(x):
    x=x.upper()
    out=[check_nt2aa(x[ind:ind+3]) for ind in range(0,len(x),3)]
    return ''.join(out)
    
def aa2int(x):
    out=[aa_int_dict[residue] for residue in x]
    return np.array(out)
    
def nt2codon(x):
    x=x.upper()
    out=[nt_codon_dict[x[ind:ind+3]] for ind in range(0,len(x),3)]
    out=np.array(out)
    return out

aa_int_dict={'A':0,
            'C':1,
            'D':2,
            'E':3,
            'F':4,
            'G':5,
            'H':6,
            'I':7,
            'K':8,
            'L':9,
            'M':10,
            'N':11,
            'P':12,
            'Q':13,
            'R':14,
            'S':15,
            'T':16,
            'V':17,
            'W':18,
            'Y':19,
            '*':20}
    
aa_keys=aa_int_dict.keys()
int_aa_dict={}
for key in aa_keys:
    curr_int=aa_int_dict[key]
    int_aa_dict[curr_int]=key
       
nt_aa_dict={'ATT':'I',
               'ATC':'I',
               'ATA':'I',
               'CTT':'L',
               'CTC':'L',
               'CTA':'L',
               'CTG':'L',
               'TTA':'L',
               'TTG':'L',
               'GTT':'V',
               'GTC':'V',
               'GTA':'V',
               'GTG':'V',
               'TTT':'F',
               'TTC':'F',
               'ATG':'M',
               'TGT':'C',
               'TGC':'C',
               'GCT':'A',
               'GCC':'A',
               'GCA':'A',
               'GCG':'A',
               'GGT':'G',
               'GGC':'G',
               'GGA':'G',
               'GGG':'G',
               'CCT':'P',
               'CCC':'P',
               'CCA':'P',
               'CCG':'P',
               'ACT':'T',
               'ACC':'T',
               'ACA':'T',
               'ACG':'T',
               'TCT':'S',
               'TCC':'S',
               'TCA':'S',
               'TCG':'S',
               'AGT':'S',
               'AGC':'S',
               'TAT':'Y',
               'TAC':'Y',
               'TGG':'W',
               'CAA':'Q',
               'CAG':'Q',
               'AAT':'N',
               'AAC':'N',
               'CAT':'H',
               'CAC':'H',
               'GAA':'E',
               'GAG':'E',
               'GAT':'D',
               'GAC':'D',
               'AAA':'K',
               'AAG':'K',
               'CGT':'R',
               'CGC':'R',
               'CGA':'R',
               'CGG':'R',
               'AGA':'R',
               'AGG':'R',
               'TAA':'*',
               'TAG':'*',
               'TGA':'*'}
               
nt_codon_dict={'ATT':0,
               'ATC':1,
               'ATA':2,
               'CTT':3,
               'CTC':4,
               'CTA':5,
               'CTG':6,
               'TTA':7,
               'TTG':8,
               'GTT':9,
               'GTC':10,
               'GTA':11,
               'GTG':12,
               'TTT':13,
               'TTC':14,
               'ATG':15,
               'TGT':16,
               'TGC':17,
               'GCT':18,
               'GCC':19,
               'GCA':20,
               'GCG':21,
               'GGT':22,
               'GGC':23,
               'GGA':24,
               'GGG':25,
               'CCT':26,
               'CCC':27,
               'CCA':28,
               'CCG':29,
               'ACT':30,
               'ACC':31,
               'ACA':32,
               'ACG':33,
               'TCT':34,
               'TCC':35,
               'TCA':36,
               'TCG':37,
               'AGT':38,
               'AGC':39,
               'TAT':40,
               'TAC':41,
               'TGG':42,
               'CAA':43,
               'CAG':44,
               'AAT':45,
               'AAC':46,
               'CAT':47,
               'CAC':48,
               'GAA':49,
               'GAG':50,
               'GAT':51,
               'GAC':52,
               'AAA':53,
               'AAG':54,
               'CGT':55,
               'CGC':56,
               'CGA':57,
               'CGG':58,
               'AGA':59,
               'AGG':60,
               'TAA':61,
               'TAG':62,
               'TGA':63}
codon_keys=nt_codon_dict.keys()
int_ntcodon_dict={}
for key in codon_keys:
    curr_int=nt_codon_dict[key]
    int_ntcodon_dict[curr_int]=key

codon_int2aa= {0:'I',
               1:'I',
               2:'I',
               3:'L',
               4:'L',
               5:'L',
               6:'L',
               7:'L',
               8:'L',
               9:'V',
               10:'V',
               11:'V',
               12:'V',
               13:'F',
               14:'F',
               15:'M',
               16:'C',
               17:'C',
               18:'A',
               19:'A',
               20:'A',
               21:'A',
               22:'G',
               23:'G',
               24:'G',
               25:'G',
               26:'P',
               27:'P',
               28:'P',
               29:'P',
               30:'T',
               31:'T',
               32:'T',
               33:'T',
               34:'S',
               35:'S',
               36:'S',
               37:'S',
               38:'S',
               39:'S',
               40:'Y',
               41:'Y',
               42:'W',
               43:'Q',
               44:'Q',
               45:'N',
               46:'N',
               47:'H',
               48:'H',
               49:'E',
               50:'E',
               51:'D',
               52:'D',
               53:'K',
               54:'K',
               55:'R',
               56:'R',
               57:'R',
               58:'R',
               59:'R',
               60:'R',
               61:'*',
               62:'*',
               63:'*'}

import matplotlib as mpl

cdict = {'red':   ((0.0,  1, 1),
                   (0.1, 1, 1),
                   (0.2,  0.65, 0.65),
                   (0.5, 1.0, 1.0),
                   (1.0,  1, 1)),

         'green': ((0.0,  1, 1),
                   (0.1, 1, 1),
                   (0.2, 0.35, 0.35),
                   (0.5, 0.35, 0.35),
                   (1.0,  1, 1)),

         'blue':  ((0.0,  1, 1),
                   (0.1,  1, 1),
                   (0.2,  0.05, 0.05),
                   (1.0,  0.05, 0.05))}

brown_yellow = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)

cdict = {'red':   ((0.0,  0.65, 0.65),
                   (0.5, 1.0, 1.0),
                   (1.0,  1, 1)),

         'green': ((0.0, 0.35, 0.35),
                   (0.5, 0.35, 0.35),
                   (1.0,  1, 1)),

         'blue':  ((0.0,  0.05, 0.05),
                   (1.0,  0.05, 0.05))}

brown_yellow2 = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)


cdict = {'red':   ((0.0,  1, 1),
                   (0.19, 1, 1),
                   (0.2,  0.65, 0.65),
                   (0.5, 1.0, 1.0),
                   (1.0,  1, 1)),

         'green': ((0.0,  1, 1),
                   (0.19, 1, 1),
                   (0.2, 0.35, 0.35),
                   (0.5, 0.35, 0.35),
                   (1.0,  1, 1)),

         'blue':  ((0.0,  1, 1),
                   (0.19,  1, 1),
                   (0.2,  0.05, 0.05),
                   (1.0,  0.05, 0.05))}

brown_yellow3 = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)

cdict = {'red':   ((0.0,  1, 0.75),
                   (0.25,  1, 0.875),
                   (0.47,  1, 1),
                   (0.53,  1, 1),
                   (0.75,  0.5, 0.5),
                   (1.0,  0.25, 0.25)),

         'green': ((0.0,  0.5, 0.5),
                   (0.25,  0.75, 0.75),
                   (0.47,  1, 1),
                   (0.53,  1, 1),
                   (0.75,  0.875, 0.875),
                   (1.0,  0.75, 0.75)),

         'blue':  ((0.0,  0, 0),
                   (0.25,  0, 0),
                   (0.47,  1, 1),
                   (0.53,  1, 1),
                   (0.75,  0.2, 0.1),
                   (1.0,  0.0, 0.0))}

red_green = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)


aa_colors = {
    'K': [0, 0, 1],
    'R': [0, 0, 1.0000],
    'H': [0, 0, 1.0000],
    'C': [0.2500, 1.0000, 0.7500],
    'F': [1.0000, 0.8125, 0],
    'G': [1.0000, 0.8125, 0],
    'A': [1.0000, 0.8125, 0],
    'I': [1.0000, 0.8125, 0],
    'L': [1.0000, 0.8125, 0],
    'M': [1.0000, 0.8125, 0],
    'N': [0.2500, 1.0000, 0.7500],
    'Q': [0.2500, 1.0000, 0.7500],
    'P': [1.0000, 0.8125, 0],
    'S': [0.2500, 1.0000, 0.7500],
    'T': [0.2500, 1.0000, 0.7500],
    'V': [1.0000, 0.8125, 0],
    'W': [1.0000, 0.8125, 0],
    'Y': [0.2500, 1.0000, 0.7500],
    'D': [1, 0, 0],
    'E': [1, 0, 0],
    '*': [0, 0, 0]}

aa_colors2 = {
    'K': [0, 0, 0.8000],
    'R': [0, 0, 0.8000],
    'H': [0, 0, 0.8000],
    'C': [0.9, 0.5, 0.2],
    'F': [1.0000, 0.8125, 0],
    'G': [1.0000, 0.8125, 0],
    'A': [1.0000, 0.8125, 0],
    'I': [1.0000, 0.8125, 0],
    'L': [1.0000, 0.8125, 0],
    'M': [1.0000, 0.8125, 0],
    'N': [0.0000, 0.8000, 0.0000],
    'Q': [0.0000, 0.8000, 0.0000],
    'P': [0.9, 0.5, 0.2],
    'S': [0.0000, 0.8000, 0.0000],
    'T': [0.0000, 0.8000, 0.0000],
    'V': [1.0000, 0.8125, 0],
    'W': [1.0000, 0.8125, 0],
    'Y': [1.0000, 0.8125, 0],
    'D': [0.8, 0, 0],
    'E': [0.8, 0, 0],
    '*': [0, 0, 0]}

mut_colors = {
    0: [0, 0, 0.8],
    1: [0, 0, 0],
    2: [0, 1, 0],
    3: [1, 0, 0],
    4: [0.5, 0, 0.5],
    5: [0.5, 0.5, 0],
    6: [0, 0.5, 0.5]}

# Only used in Fig. 4. 
# ['P','E', 'V','N','L']

#syn_colors = {  'E':[0,1,0], # Glutamic acid, Green 
#                'P':[1,1,0], # Proline, Yellow
#                'L':[1,0,1], # Leucine, Magenta
#                'N':[1,.5,0],# Asparagine, Orange
#                'V':[0,1,1]} # Cysteine, Cyan

syn_colors = {'R':[0.5,0.5,1],
               'H':[0,0,1],
               'Y':[0,0.8,0],
               'S':[0,0.0,0.8],
               'D':[0.8,0,0],
               'E':[1,0.5,0.5],
               'G':[0,0.5,0.2],
               'P':[1,0.8,0.2],
               'A':[0.2,0.2,0.2],
               'C':[0.0,1,0.3],
               'F':[0,0.5,0],
               'I':[0.4,0,0.4],
               'L':[0.7,0,0.7],
               'M':[0.7,0.7,0.7],
               'N':[1,0,1],
               'Q':[1,0.4,1],
               'T':[0.3,1,0],
               'V':[0,0.8,0.8],
               'W':[0.2,0.2,0.2],
               'K':[0,0,0.7]}