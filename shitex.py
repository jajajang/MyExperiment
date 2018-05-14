import torch as th
from nltk.corpus import wordnet as wn
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