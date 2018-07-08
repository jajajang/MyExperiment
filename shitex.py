import torch as th
from nltk.corpus import wordnet as wn
import numpy as np
z_dim=5
indy_mix=[]
ordered_word=[]
f=open('wnids.txt','r')
while True:
    line=f.readline()
    if not line: break
    #ordered_word.append(wn.synset_from_pos_and_offset(line[0],int(line[1:-1])).name())
    indy_mix.append(int(line[1:-1]))

ordered_indy=sorted(indy_mix)
for i in range(0,len(ordered_indy)):
    ordered_word.append(wn.synset_from_pos_and_offset('n',ordered_indy[i]).name())

tempy=th.load('dim_5_200.pth')
obj=tempy['objects']
position_=(tempy['model']['lt.weight']).float()

true_pos=th.zeros(200,z_dim)
for i in range(0,200):
    true_pos[i]=position_[obj.index(ordered_word[i])]

zetty=-th.sum(true_pos*true_pos, dim=-1).expand(z_dim,-1)+1
position=2*true_pos/zetty.t()
position_zet= (th.sqrt(th.sum(position*position, dim=-1)+1))


zetty_all=(-th.sum(position_*position_, dim=-1).expand(z_dim,-1)+1)
position_all=2*position_/zetty_all.t()
position_all_zet= (th.sqrt(th.sum(position_all*position_all, dim=-1)+1))


synsetss=[0]*len(obj)
for i in range(0,len(obj)):
    synsetss[i]=wn.synset(obj[i])


level=[0]*len(obj)
for i in range(0,len(obj)):
    pathy=synsetss[i].hypernym_paths()[0]
    for j in range(0,len(obj)):
        if synsetss[j] in pathy:
            level[i]+=1

bitbit=[0]*12
for i in range(2,12):
    bitbit[i]=[j for j,x in enumerate(level) if x==i]

ordy=[0]*200
for i in range(0,200):
    ordy[i]=wn.synset(ordered_word[i])

categorize=[0]*12
for i in range(0,12):
    categorize[i]=[0]*200

for s in range(2,12):
    for i in range(0,200):
        pathy=ordy[i].hypernym_paths()[0]
        tik=True
        for j in bitbit[s]:
            if synsetss[j] in pathy:
                categorize[s][i]=j
                tik=False
            if tik:
                categorize[s][i]=categorize[s-1][i]    

for s in range(2,12):
    level_sum=0
    for i in range(0,200):
        x=position_all_zet[categorize[s][i]]
        level_sum+=np.log(x+np.sqrt(x*x+1))
    print level_sum/200
