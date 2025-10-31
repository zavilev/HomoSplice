import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from Bio import SeqIO
from Bio.Seq import Seq
import subprocess
import warnings
warnings.simplefilter('ignore')
from itertools import combinations
from Bio.Align import substitution_matrices

from gtfparse import read_gtf

from pymsaviz import MsaViz
from Bio.Align import MultipleSeqAlignment as MSA, SeqRecord
import matplotlib as mpl
import networkx as nx
import matplotlib.patheffects as pe

import os
import glob
import contextlib
from io import StringIO
from collections import defaultdict

def desc2dict(desc):
    desc = desc.split("; ")
    dic = defaultdict(list)
    for item in desc:
        item = item.split(" ")
        if len(item)==2:
            dic[item[0]].append(item[1].strip('";'))
    return {key:",".join(value) for key,value in dic.items()}

def load_annotation(gtfile,chunksize=1000000):
    res = []
    for gtf in pd.read_csv(gtfile,header=None,sep='\t', comment='#',
                           names=["chrn","source","type","start",
                                  "end",'score',"strand",'frame',"desc"],chunksize=chunksize):

        desc = gtf.apply(lambda x: desc2dict(x.desc), axis=1, result_type="expand")
        gtf = pd.concat([gtf,desc],axis=1).drop('desc',axis=1)
        res.append(gtf.copy())
    res = pd.concat(res)
    res[['start','end']] = res[['start','end']].astype(int)
    return res

class Sequence():
    GENOMES = dict()
    GTFS = dict()
    @classmethod
    def add_genome(cls,name='human',mask='/gss/dplab/data/common/genomes/GRCh38/annotation/GRCh38/GRCh38.fa',
                   addchr=True, override=False,obj=None,singlefasta=True): #path to chrom fasta files
        if obj is not None:
            Sequence.GENOMES[name] = obj
            return
        if (name not in Sequence.GENOMES.keys()) or override:
            if singlefasta:
                Sequence.GENOMES[name] = SeqIO.to_dict(SeqIO.parse(mask, "fasta"))
                keys = list(Sequence.GENOMES[name].keys())
                for key in keys:
                    if '_' in key:
                        Sequence.GENOMES[name].pop(key, None)
                    else:
                        Sequence.GENOMES[name][key] = Sequence.GENOMES[name][key].seq
            else:
                Sequence.GENOMES[name] = dict()
                for file in glob.glob(mask):
                    chrn = SeqIO.read(file, 'fasta')
                    key = chrn.id if not addchr else f'chr{chrn.id}'
                    Sequence.GENOMES[name][key] = chrn.seq
    @classmethod
    def add_gtf(cls,name='human',
                mask='/gss/home/l.zavileisky.ext/TCGA/GRCh38_dna/annotation/Homo_sapiens.GRCh38.108.chr.chromnames.gtf',
                override=False,obj=None,filtering=True,use_gtfparse=True):
        if obj is not None:
            Sequence.GTFS[name] = obj
            return
        if (name not in Sequence.GTFS.keys()) or override:
            if use_gtfparse:
                an = read_gtf(mask,result_type='pandas').rename(columns={'seqname':'chrn','feature':'type'})
                an[['start','end']] = an[['start','end']].astype(int)
            else:
                an = load_annotation(mask)
            an['chrn'] = an.chrn.astype(str)
            if 'transcript_biotype' in an.columns:
                an = an[an.transcript_biotype.isin(['protein_coding','nonsense_mediated_decay'])]
            else:
                an['transcript_biotype'] = 'undefined'
            if filtering:
                all_transcripts = an[an.type.isin(['transcript','exon','start_codon','stop_codon'])].groupby('type').transcript_id.agg(set)
                u1 = set.intersection(*all_transcripts)
                an = an[an.transcript_id.isin(u1)]
            Sequence.GTFS[name] = an
        
    def __init__(self,name,chrn,strand,borders,genome):
        self.name = name
        self.chrn = chrn
        self.strand = strand
        self.borders = borders
        self.genome = genome
    @staticmethod
    def from_gtf(gene,genome='human',gtf='human',name=None):
        an = Sequence.GTFS[gtf]
        if 'tag' not in an.columns:
            an['tag'] = ''
        Genome = Sequence.GENOMES[genome]
        df = an[(an.gene_name==gene)&(an.type=='CDS')].copy()
        if len(df)==0:
            print(f'Gene "{gene}" not found in gtf "{gtf}"')
            return None
        df.start-=1
        borders = sorted(df[['start','end']].melt().value.unique())
        chrn = df.chrn.iloc[0]
        strand = df.strand.iloc[0]
        msa = Sequence.gtf2msa(df,borders,chrn,strand,Genome)

        ind = msa.index.tolist()
        aamsa = msa.apply(Sequence.nt2aa).fillna('').applymap(lambda x: str(Seq(x).translate())).transpose()[ind]
        cols = aamsa.columns
#        if strand=='-':
#            cols = cols[::-1]
        aamsa = aamsa[cols]
        
        transcripts = aamsa.index
        tw = Sequence.getweights(transcripts,an)
    
        aaseq = aamsa.apply(Sequence.select_best_aa,args=(tw,))
        aaseq = aaseq[aaseq!='']
        
        borders = Sequence.get_borderinfo(df,borders)
        borders = borders.loc[aaseq.index]
        borders['curr'] = borders.index
        borders['prev'] = [np.nan]+borders.index.tolist()[:-1]
        borders['delta'] = abs(borders.curr-borders.prev)
        borders['exon_group'] = np.cumsum((borders.delta!=1).astype(int))
        borders.drop(['curr','prev','delta'],axis=1,inplace=True)
        borders['seq'] = aaseq
        if name is None:
            name = gene
        return Sequence(name,chrn,strand,borders,genome)

    @staticmethod
    def gtf2msa(df,borders,chrn,strand,genome):
        msa = pd.DataFrame()
        for i in range(len(borders)-1):
            s,e = borders[i],borders[i+1]
            seq = genome[chrn][s:e]
            if strand=='-':
                seq = seq.reverse_complement()
            seq = str(seq)
            for tr in df[(df.start<=s)&(df.end>=e)].transcript_id.unique():
                msa.loc[i,tr] = seq
        if strand=='-':
            msa = msa.loc[::-1]
        return msa

    @staticmethod
    def nt2aa(col):
        x = col[col.notna()].copy()
        extra = ''
        for i in range(len(x)):
            ex = len(x.iloc[i])%3
            if ex==1:
                extra = x.iloc[i][-1]
                x.iloc[i] = x.iloc[i][:-1]
                x.iloc[i+1] = extra + x.iloc[i+1]
            if ex==2:
                extra = x.iloc[i+1][0]
                x.iloc[i+1] = x.iloc[i+1][1:]
                x.iloc[i] = x.iloc[i] + extra
        return x

    @staticmethod
    def getweights(transcripts,an,weights=None):
        if weights is None:
            weights = {'MANE_Plus_Clinical':10,
                       'MANE_Select':100,
                       'basic':5,
                       'mRNA_end_NF':-5,
                       'mRNA_start_NF':-5}

        tw = an[(an.type=='transcript')&
                (an.transcript_id.isin(transcripts))][['transcript_id','tag']].fillna('')
        tw['weight'] = 0
        for key,val in weights.items():
            tw['weight']+= tw.tag.str.contains(key).astype('int')*val
        return tw.set_index('transcript_id')['weight']

    @staticmethod
    def select_best_aa(col,tw):
        tw1 = tw.loc[col[(col!='')&(col.notna())].index]
        if len(tw1)==0:
            return ''
        tr = tw1[tw1==max(tw1)].sample(1).index[0]
        return col.loc[tr]

    @staticmethod
    def get_borderinfo(df,borders):
        borders = pd.DataFrame({'istart':borders[:-1],'iend':borders[1:]}).reset_index()
        m = df[['start','end','tag','transcript_id','transcript_biotype']].merge(borders,how='cross')
        m = m[(m.start<=m.istart)&(m.iend<=m.end)]
        
        m['mane'] = m.tag.fillna('').str.contains('MANE_Select').astype('int')
        m = m.groupby(['index','istart','iend']).agg({'mane':max,
                                                      'transcript_id':set,
                                                      'transcript_biotype':set})
        m['trs'] = m.transcript_id.apply(len)
        m = m.rename(columns={'transcript_biotype':'biotype'}).reset_index()
        m.biotype = m.biotype.apply(lambda x: '_'.join(sorted(x)))
        m.biotype = m.biotype.map({'nonsense_mediated_decay_protein_coding':'both',
                                   'nonsense_mediated_decay':'nmd', 'protein_coding':'coding'}).fillna('other')
        return m.set_index('index')
    
    
class Family():
    def __init__(self,lst=[]):
        self.dict = {i.name:i for i in lst}
        self.msa = None
        self.msa_borders = None
        self.msa_score = None
    def add_sequence(self,seq):
        self.dict[seq.name] = seq
    def add_from_gtf(self,gene,gtf='human',genome='human',name=None,multiexon=True):
        try:
            seq = Sequence.from_gtf(gene,genome,gtf,name)
            if seq is not None:
                if (not multiexon) or (len(seq.borders.exon_group.unique())>1):
                    self.add_sequence(seq)
        except Exception as e:
            print(f'Error while adding gene "{gene}" from gtf "{gtf}":')
            print(e)
    def align(self,cmd=None):
        fasta = []
        for key,val in self.dict.items():
            fasta.append('>'+key)
            fasta.append("".join(val.borders.seq))
        fasta = '\n'.join(fasta)

        if cmd is None:
            src = r'"C:\Program Files (x86)\clustal-omega-1.2.2-win64\clustalo.exe"'
            cmd = f'{src} -i - --outfmt fa --wrap {1000000}'
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate(input=fasta.encode())
        maf = res[0].decode()
        if not maf:
            print(res[1].decode())
            return

        res = dict()
        for rec in maf[1:].split('>'):
            key,seq = rec.strip().split('\n')
            seq = list(seq.strip())
            key = key.strip()
            res[key] = seq
        self.msa = pd.DataFrame(res)

    def _msa2bordernum(self,borders=True):
        if self.msa is None:
            self.align()
        results = self.msa.copy()
        for col in results.columns:
            nums = []
            d = self.dict[col].borders.copy()
            d['l'] = d.seq.apply(len)
            if borders:
                d = d.l
            else:
                d = d.groupby('exon_group').l.agg(sum)
            for num,l in d.items():
                nums+=[num]*l
            results.loc[results[col]!='-',col] = nums
        #inner gaps are treated as exons
        for c in results.columns:
            col = results[c]
            innergaps = col[col!='-'].reset_index().groupby(c)['index'].agg([min,max])
            for num,row in innergaps.iterrows():
                results.loc[row['min']:row['max'],c] = num
        return results

    def msa2bordernum(self):
        self.msa_borders = self._msa2bordernum(borders=True)

    def get_scores(self, matrix="BLOSUM62",smooth=5,gapval=0.5):
        mat = substitution_matrices.load(matrix)
        ind = list(mat.alphabet)
        mat = pd.DataFrame(mat,index=ind,columns=ind)#.rename(columns={'*':'-'},index={'*':'-'})
        mat['-'] = -4
        mat.loc['-'] = -4
        mat.loc['-','-'] = gapval
        
        def _scores(ind1,ind2,mat):
            return [mat.loc[i1,i2] for i1,i2 in zip(ind1,ind2)]

        if self.msa is None:
            self.align()
        colscore = dict()
        for i,j in combinations(self.msa.columns,2):
            colscore[(i,j)] = pd.Series(_scores(self.msa[i],self.msa[j],mat))
        #self.pairwise_msa_scores = pd.DataFrame(colscore)
        colscore = pd.DataFrame(colscore)
        colscore['score'] = colscore.apply(np.mean,1)
        def slidingsmooth(x,size):
            if size<1:
                return x
            res = []
            for i in range(1,size+1):
                res.append(np.mean(x[:i]))
            for i in range(size+1,len(x)+1):
                res.append(np.mean(x[i-size:i]))
            return res
        colscore.score = slidingsmooth(colscore.score,smooth)
        self.msa_score = colscore


    def _to_attraction(self,pairwise=True,borders=True):
        res = []
        borderdf = self._msa2bordernum(borders)
        if self.msa_score is None:
            self.get_scores()
        for s1,s2 in combinations(self.msa.columns,2):
            df = borderdf[[s1,s2]]
            if pairwise:
                df['score'] = self.msa_score[s1][s2]
            else:
                df['score'] = self.msa_score.score
            condition = (self.msa[s1]!='-')|(self.msa[s2]!='-')
            df['len'] = ((df[s1]!='-')&(df[s1]!='-')).astype('int')
            df['pos'] = df.index
            #len is length of alignment without gaps
            for (e1,e2),tmp in df[condition].groupby([s1,s2]):
                #clip gaps at the edge of exons
                tmp2 = self.msa.loc[tmp.index]
                cond = (tmp2[s1]!='-')&(tmp2[s2]!='-')
                cond = cond[cond]
                if len(cond)!=0:
                    left,right = cond[cond].index[0],cond[cond].index[-1]
                    tmp = tmp.loc[left:right]
                    res.append([s1,s2,e1,e2,sum(tmp['score']),sum(tmp['len']),np.mean(tmp.pos)])
        res = pd.DataFrame(res,columns=['s1','s2','b1','b2','sum','len','pos'])
        return res[(res.b1!='-')&(res.b2!='-')&(res['sum']>0)&(res['len']>1)]
    
    def _to_attraction_depr(self,pairwise=True,borders=True):
        res = []
        borderdf = self._msa2bordernum(borders)
        if self.msa_score is None:
            self.get_scores()
        for s1,s2 in combinations(self.msa.columns,2):
            df = borderdf[[s1,s2]]
            if pairwise:
                df['score'] = self.msa_score[s1][s2]
            else:
                df['score'] = self.msa_score.score
            condition = ~((self.msa[s1]=='-')&(self.msa[s2]=='-'))
            df['len'] = ((df[s1]!='-')&(df[s1]!='-')).astype('int')
            df['pos'] = df.index
            #len is length of alignment without gaps
            df = df[condition].groupby([s1,s2]).agg({'score':sum,
                                                     'len':sum,
                                                     'pos':np.mean}).reset_index().rename(columns={s1:'b1',
                                                                                          s2:'b2','score':'sum'})
            df = df[(df.b1!='-')&(df.b2!='-')&(df['sum']>0)&(df['len']>1)]
            df['s1'] = s1
            df['s2'] = s2
            res.append(df)
        return pd.concat(res)


    def to_attraction(self,pairwise=True,borders=True):
        res = self._to_attraction(pairwise,borders)
        res2 = res.rename(columns={'b1':'b2','b2':'b1','s1':'s2','s2':'s1'})
        res = pd.concat([res,res2])
        res['key'] = res[['s2','b2']].apply(tuple,1)
        edges = dict()
        for key,tmp in res.groupby(['s1','b1']):
            edges[key] = tmp.set_index('key')['sum'].to_dict()
        return edges

    @staticmethod
    def _borders2graph(tmp,ilength):
        df = tmp[['istart','iend','exon_group']].reset_index()
        df['l'] = df.iend-df.istart
        el = df.groupby('exon_group').l.agg(sum)
        rpos = np.cumsum(el+ilength)
        lpos = rpos-el
        df['exon_lpos'] = df.exon_group.map(lpos)
        df['exon_rpos'] = df.exon_group.map(rpos)
        df['exon_length'] = df.exon_rpos-df.exon_lpos
        
        df['cuml'] = df.groupby('exon_group').l.transform(np.cumsum)
        df['border_lpos'] = df.exon_lpos+df.cuml-df.l
        df['border_rpos'] = df.exon_lpos+df.cuml
        
        df = df[['index','exon_group','exon_lpos','exon_rpos',
                 'border_lpos','border_rpos','l','exon_length']]
        
        df['ex'] = (df.exon_lpos+df.exon_rpos)/2
        df['bx'] = (df.border_lpos+df.border_rpos)/2
        df['dist'] = df.bx-df.ex
        
        leftmost = df[['index','exon_group']].groupby('exon_group').head(1).rename(columns={'index':'left'})
        rightmost = df[['index','exon_group']].groupby('exon_group').tail(1).rename(columns={'index':'right'})
        leftmost.exon_group-=1
        repulsion = rightmost.merge(leftmost)
    
        rep1 = repulsion[['right','left']]
        rep2 = rep1.copy()
        rep2.columns = list(rep1.columns)[::-1]
        rep = pd.concat([rep1,rep2]).groupby('left').right.agg(tuple)
        for ind in set(df['index'])-set(rep.index):
            rep.loc[ind] = tuple()
        df['repulsion'] = df['index'].map(rep)
        return df
    
    @staticmethod
    def _2border(row):
        repulsion = {(row.seq,i) for i in row.repulsion}
        return Border(row.dist,row.l,row.seq,row.exon_group,row['index'],repulsion)
    
    @staticmethod
    def _2node(row):
        return Node(row.ex,row.exon_length,row.seq,row.exon_group,row.key)
    
    def to_graph(self,ilength=200,pairwise=False):
        res = []
        for key,tmp in self.dict.items():
            df = Family._borders2graph(tmp.borders,ilength)
            df['seq'] = key
            res.append(df)
        res = pd.concat(res)
        res['obj'] = res.apply(lambda x: Family._2border(x),axis=1)
        res['key'] = res[['seq','index']].apply(tuple,1)
        bdict = res.set_index('key').obj.to_dict()
        ndict = res.groupby(['seq','exon_group','ex','exon_length']).key.agg(list).reset_index()
        ndict['obj'] = ndict.apply(Family._2node,1)
        ndict = ndict.set_index(['seq','exon_group']).obj.to_dict()
        attraction = self.to_attraction(pairwise)
        for key,val in attraction.items():
            bdict[key].attraction = val
        return Graph(ndict,bdict,ilength)
    def to_table(self):
        #returns table of exon and intron genomic coordinates
        old = []
        for seq,obj in self.dict.items():
            df = obj.borders.groupby('exon_group').agg({'istart':min,'iend':max}).reset_index()
            df['seqname'] = seq
            df['chrn'] = obj.chrn
            df['strand'] = obj.strand
            df['type'] = 'exon'
            introns = df.copy()
            if obj.strand=='+':
                introns['prev'] = [np.nan]+introns.iend.tolist()[:-1]
                introns = introns[introns.prev.notna()].drop('iend',1).rename(columns={'prev':'istart',
                                                                                       'istart':'iend'})
            else:
                introns['prev'] = [np.nan]+introns.istart.tolist()[:-1]
                introns = introns[introns.prev.notna()].drop('istart',1).rename(columns={'prev':'iend',
                                                                                       'iend':'istart'})
    
            introns['type'] = 'intron'
            df = pd.concat([df,introns])[['seqname', 'chrn', 'strand', 'exon_group',  'type', 'istart', 'iend']]
            df.istart = df.istart.astype('int')
            old.append(df)
        old = pd.concat(old)
        return old
    
    def msa2plot_coords(self,G=None,adjust=True,G_kws=dict(ilength=200,pairwise=True),adjust_kws=dict(iterations=200)):
        if G is None:
            G = self.to_graph(**G_kws)
        if adjust:
            G.FD(**adjust_kws)
        
        mask = self.msa=='-'
        df = self.msa.copy()
        df.loc[:,:] = 3
        df = df.where(~mask,other=0).apply(np.cumsum)-2
        
        tab = self.to_table()
        tab['l'] = abs(tab.iend-tab.istart)
        tabe = tab[tab.type=='exon'].drop('type',1)
        tabi = tab[tab.type=='intron'].drop(['type','istart','iend'],1).rename(columns={'l':'l_i'})
        tab = tabe.merge(tabi,how='left').fillna(0)
        tab = tab.sort_values('exon_group')
        tab['cum_l'] = tab.groupby('seqname').l.transform(np.cumsum)-tab.l
        
        for _,row in tab.loc[::-1].iterrows():
            df.loc[df[row.seqname]>=row.cum_l,row.seqname]+=row.l_i
        df = df.where(~mask,other=np.nan)
        
        old = self.to_table()
        new = G.to_table()[1]
        ctab = old.merge(new)
        ctab.loc[(ctab.strand=='-'),
                 ['estart','eend']] = ctab.loc[(ctab.strand=='-'),
                                               ['estart','eend']].rename(columns={'estart':'eend','eend':'estart'})
        
        res = df.copy()
        for col in df.columns:
            chrn = self.dict[col].chrn
            strand = self.dict[col].strand
            if strand == "+":
                start = self.dict[col].borders.iloc[0].istart
                res[col]+=start
            else:
                start = self.dict[col].borders.iloc[0].iend
                res[col] = start - res[col]
            lst = [(col,x) for x in res[col]]
            res[col] = convert_coords(lst,tab=ctab)
        return G,res

    def exon_homology(self,lthr=3,sthr=10,pairwise=True,edges=None):
        nodes = set()
        for seq,val in self.dict.items():
            for ex in val.borders.exon_group.unique():
                nodes.add((seq,ex))
        if edges is None:
            edges = self._to_attraction(borders=False,pairwise=pairwise)
            edges = edges[(edges['len']>lthr)&(edges['sum']>sthr)]
            edges['ratio'] = edges['sum']/edges['len']
            edges = edges.loc[~((edges['len']<10)&(edges.ratio>3))]
    
        edges['source'] = edges[['s1','b1']].apply(tuple,1)
        edges['target'] = edges[['s2','b2']].apply(tuple,1)
        
        G = nx.from_pandas_edgelist(edges)
        compdf = []
        i=0 #if no connected components present
        for i,comp in enumerate(nx.connected_components(G)):
            for key in comp:
                chrn,strand = self.dict[key[0]].chrn,self.dict[key[0]].strand
                df = self.dict[key[0]].borders
                df = df[df.exon_group==key[1]]
                l,r = min(df.istart),max(df.iend)
                compdf.append([*key,chrn,strand,l,r,i,0])
                nodes.remove(key)
        for j,key in enumerate(sorted(nodes)):
            chrn,strand = self.dict[key[0]].chrn,self.dict[key[0]].strand
            df = self.dict[key[0]].borders
            df = df[df.exon_group==key[1]]
            l,r = min(df.istart),max(df.iend)
            compdf.append([*key,chrn,strand,l,r,j+1+i,1])
        ehom = pd.DataFrame(compdf,columns=['seqname','exon_group','chrn',
                                            'strand','start','end','homogroup','alone'])
        #rename homogoups in order of exons
        hnum = ehom.groupby('homogroup').exon_group.agg(np.mean).sort_values().reset_index()
        hnum.exon_group = 1
        hnum.exon_group = np.cumsum(hnum.exon_group)-1
        rename = hnum.set_index('homogroup').exon_group
        ehom.homogroup = ehom.homogroup.map(rename)
        return ehom


    def intron_homology(self,lthr=3,sthr=10,pairwise=True,edges=None):
        if edges is None:
            edges = self._to_attraction(borders=False,pairwise=pairwise)
            edges = edges[(edges['len']>lthr)&(edges['sum']>sthr)]
            edges['ratio'] = edges['sum']/edges['len']
            edges = edges.loc[~((edges['len']<10)&(edges.ratio>3))]
        e2 = edges.rename(columns={'b1':'b2','b2':'b1','s1':'s2','s2':'s1'})
        edges = pd.concat([edges,e2])
        edges = edges.sort_values('pos')
    
        r1 = edges.groupby(['s1','b1','s2']).tail(1)
        r2 = edges.groupby(['s2','b2','s1']).tail(1)
        r = r1.merge(r2)
        
        #get reciprocal leftmost edges
        l1 = edges.groupby(['s1','b1','s2']).head(1)
        l2 = edges.groupby(['s2','b2','s1']).head(1)
        l = l1.merge(l2)
    
        r = Family.edges2introngroups(r)
        r = r.groupby(['seq','hom']).tail(1)
        l = Family.edges2introngroups(l)
        l = l.groupby(['seq','hom']).head(1)
        l.num-=1
    
        r['type'] = 'r'
        l['type'] = 'l'
        rl = pd.concat([r,l])
        maxn = {k:v.borders.exon_group.iloc[-1] for k,v in self.dict.items()}
        rl['maxn'] = rl.seq.map(maxn)
        rl = rl[(rl.num>0)&(rl.num<rl.maxn)].drop('maxn',1)
    
    
        rl['node'] = rl[['seq','num']].apply(tuple,1)
        G = nx.Graph()
        for _,grp in rl.groupby(['type','hom']):
            h = nx.complete_graph(grp.node)
            G = nx.compose(G,h)
        compdf = []
        for i,comp in enumerate(nx.connected_components(G)):
            df = pd.DataFrame(list(comp),columns=['seqname','inum'])
            df['ihomogroup'] = i
            compdf.append(df)
        compdf = pd.concat(compdf).sort_values(['ihomogroup','seqname','inum']).reset_index(drop=True)
    
        func = lambda x: Family.get_coordinates(self,x.seqname,x.inum)
        compdf[['chrn','strand','start','end']] = pd.DataFrame(compdf.apply(func,1).tolist())

        #rename homogoups in order of introns
        hnum = compdf.groupby('ihomogroup').inum.agg(np.mean).sort_values().reset_index()
        hnum.inum = 1
        hnum.inum = np.cumsum(hnum.inum)-1
        rename = hnum.set_index('ihomogroup').inum
        compdf.ihomogroup = compdf.ihomogroup.map(rename)
        return compdf

    def intron_homology_depr(self,lthr=3,sthr=10,pairwise=True,edges=None):
        if edges is None:
            edges = self._to_attraction(borders=False,pairwise=pairwise)
            edges = edges[(edges['len']>lthr)&(edges['sum']>sthr)]
            edges['ratio'] = edges['sum']/edges['len']
            edges = edges.loc[~((edges['len']<10)&(edges.ratio>3))]
        e2 = edges.rename(columns={'b1':'b2','b2':'b1','s1':'s2','s2':'s1'})
        edges = pd.concat([edges,e2])
        edges = edges.sort_values('pos')
    
        r1 = edges.groupby(['s1','b1','s2']).tail(1)
        r2 = edges.groupby(['s2','b2','s1']).tail(1)
        r = r1.merge(r2)
        
        #get reciprocal leftmost edges
        l1 = edges.groupby(['s1','b1','s2']).head(1)
        l2 = edges.groupby(['s2','b2','s1']).head(1)
        l = l1.merge(l2)
        
        r = Family.edges2introngroups(r)
        l = Family.edges2introngroups(l)
        #r.num-=1
        l.num-=1
        
        r['type'] = 'r'
        l['type'] = 'l'
        rl = pd.concat([r,l])
        maxn = {k:v.borders.exon_group.iloc[-1]-1 for k,v in self.dict.items()}
        rl['maxn'] = rl.seq.map(maxn)
        rl = rl[(rl.num>=0)&(rl.num<rl.maxn)].drop('maxn',1)
        rl['node'] = rl[['seq','num']].apply(tuple,1)
        G = nx.Graph()
        for _,grp in rl.groupby(['type','hom']):
            h = nx.complete_graph(grp.node)
            G = nx.compose(G,h)
        compdf = []
        for i,comp in enumerate(nx.connected_components(G)):
            df = pd.DataFrame(list(comp),columns=['seqname','inum'])
            df['ihomogroup'] = i
            compdf.append(df)
        compdf = pd.concat(compdf).sort_values(['ihomogroup','seqname','inum']).reset_index(drop=True)
        compdf.inum+=1
        func = lambda x: Family.get_coordinates(self,x.seqname,x.inum)
        compdf[['chrn','strand','start','end']] = pd.DataFrame(compdf.apply(func,1).tolist())
        return compdf

    @staticmethod
    def edges2introngroups(r):
        '''
        1. get connected components from edgelist
        2. filter each component so as only one exon per sequence is present
        '''
        r['source'] = r[['s1','b1']].apply(tuple,1)
        r['target'] = r[['s2','b2']].apply(tuple,1)
        G = nx.from_pandas_edgelist(r)
        compdf = []
        for i,comp in enumerate(nx.connected_components(G)):
            df = pd.DataFrame(list(comp),columns=['seq','num'])
            df['hom'] = i
            compdf.append(df)
        compdf = pd.concat(compdf).sort_values(['hom','seq','num'])
        return compdf

    @staticmethod
    def get_coordinates(fam,seq,num):
        df = fam.dict[seq].borders
        left = df[df.exon_group==num]
        right = df[df.exon_group==num+1]
        if fam.dict[seq].strand=='+':
            l = left.iend.iloc[-1]
            r = right.istart.iloc[0]
        else:
            r = left.istart.iloc[0]
            l = right.iend.iloc[-1]
        return fam.dict[seq].chrn,fam.dict[seq].strand,l,r


    def draw_msa(fam,G=None,order=None,width=0.2,highlight=None,figsize=(10,4),
                 coloring = None,alpha=0.5,enum=True,colorthr = 0):
        adjust=True
        if G is not None:
            adjust=False
        G,res = fam.msa2plot_coords(G=G,adjust=adjust,G_kws={'ilength': 200, 'pairwise': True},adjust_kws=dict(iterations=500,t=10))
        blosum_cmap = plt.get_cmap('vlag')
        if order is None:
            order = res.columns
        order = {j:i for i,j in enumerate(order)}
        
        tabs = G.to_table()[0]
        tabs['y'] = tabs.seqname.map(order)
        tabsc = tabs.copy()
        
        borders = []
        for seq,val in fam.dict.items():
            df = val.borders.copy().drop(['seq','istart','iend'],1)
            df['ind'] = df.index
            df['seqname'] = seq
            borders.append(df)
        borders = pd.concat(borders)
        tabs = tabs.drop(['estart','eend'],1).merge(borders)
    
        score = fam.msa_score.to_numpy().flatten()
        vmin,vmax = min(score),max(score)
        if vmin>=0:
            blosum_norm = mpl.colors.Normalize(-vmax,vmax)
        elif vmax<=0:
            blosum_norm = mpl.colors.Normalize(vmin,-vmin)
        else:
            blosum_norm = mpl.colors.TwoSlopeNorm(0,vmin,vmax)
        
        cmap = plt.get_cmap('tab10')
        if coloring is None:
            bcolor = {'both':'grey','coding':cmap(0),'nmd':cmap(3),'other':cmap(9)}
            tabs['bcolor'] = tabs.biotype.map(bcolor)
        else:
            tabs['bcolor'] = tabs[['seqname','exon_group']].apply(tuple,1).map(coloring).fillna('black')
        
        get_color = lambda x: cmap(x%9+1)
        if highlight is not None:
            if 'mane' in highlight:
                high = set(highlight)-{'mane'}
                highlight = dict(mane = cmap(0))
                for i,j in enumerate(high):
                    highlight[j]=get_color(i)
            else:
                highlight = {j:get_color(i) for i,j in enumerate(highlight)}
        
        f,(ax,cax) = plt.subplots(1,2,figsize=figsize,width_ratios=[19,1],constrained_layout=True)
        for _,row in tabs.iterrows():
            ax.fill_between([row.bstart,row.bend],row.y-width,row.y+width,color=row.bcolor,alpha=alpha,edgecolor='black',lw=0.4)
        if enum:
            middle = tabsc.groupby(['exon_group','y']).agg({'estart':min,'eend':max})
            middle = (middle.eend+middle.estart)/2
            for (num,y),x in middle.items():
                ax.text(x,y-width,num,ha='center',va='bottom',color='black',zorder=10)
                #ax.text(x,y,num,color='white',zorder=10,ha='center',va='center',
                #        path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax.set_yticks(list(order.values()),list(order.keys()))    
        
        def get_width(number,y,width,space=1):
            trwidth = 2*width/(number*space+space+number)
            space*=trwidth
            coords = [space+trwidth/2+(trwidth+space)*i-width+y for i in range(number)]
            return coords,trwidth/2
        
        if highlight is not None:
            for y,df in tabs.groupby('y'):
                transcripts = [i for i in highlight.keys() if (i in set.union(*df.transcript_id))|(i=='mane')]
                ycoords,trw = get_width(len(transcripts),y,width)
                for yc,tr in zip(ycoords,transcripts):
                    color = highlight[tr]
                    if tr=='mane':
                        subset = df[df.mane==1]
                    else:
                        subset = df[df.transcript_id.apply(lambda x: tr in x)]
                    for _,row in subset.iterrows():
                        ax.fill_between([row.bstart,row.bend],yc-trw,yc+trw,color=color,alpha=1,edgecolor='black',lw=0.4)
        ax.invert_yaxis()
    
        cols = pd.Series(order).sort_values().index.tolist()
        for i in range(len(cols)-1):
            col1 = cols[i]
            col2 = cols[i+1]
            y1 = order[col1]+width*1.2
            y2 = order[col2]-width*1.2
            colkey = (col1,col2)
            if colkey not in fam.msa_score.columns:
                colkey = (col2,col1)
            for index,row in res[(res[col1].notna())&(res[col2].notna())].iterrows():
                x1,x2 = row[col1],row[col2]
                color = fam.msa_score.loc[index,colkey]
                if color>colorthr:
                    color = blosum_cmap(blosum_norm(color))
                    ax.fill_betweenx([y1,y2],[x1,x2],[x1+1,x2+1],color=color,alpha=0.5)
        mpl.colorbar.ColorbarBase(cax, cmap=blosum_cmap, norm=blosum_norm)
        return f,G


    def msaviz(self,ind=None,sequences=None,
                       color_scheme='Identity', wrap_length=80, show_grid=True, 
                       show_consensus=True,cmap=plt.get_cmap('tab10')):
        textann = []
        if sequences is None:
            sequences = self.dict.keys()
        if ind is None:
            index = self.msa.index
        elif isinstance(ind,dict):
            index = set()
            df = self._msa2bordernum(borders=False)
            for key,val in ind.items():
                if isinstance(val,int):
                    addind = df.loc[df[key]==val].index
                    index|=set(addind)
                    textann.append([key,val,addind[0],addind[-1]])
                else:
                    for v in val:
                        addind = df.loc[df[key]==v].index
                        index|=set(addind)
                        textann.append([key,v,addind[0],addind[-1]])
            index = sorted(index)
        else:
            index = list(ind)
        msa = []
        for seq in sequences:
            rec = ''.join(self.msa.loc[index,seq])
            msa.append(SeqRecord(rec,id=seq))
        msa = MSA(msa)
        mv = MsaViz(msa, color_scheme=color_scheme, wrap_length=wrap_length, 
                    show_grid=show_grid, show_consensus=show_consensus)
        for i,ann in enumerate(textann):
            color = cmap(i%10)
            l = index.index(ann[2])+1
            r = index.index(ann[3])+1
            mv.add_text_annotation((l,r), f"{ann[0]}:{ann[1]}", text_color=color, range_color=color)
        f = mv.plotfig()
        return f,mv

    def candidate_missing_exons(self,lthr=3,sthr=10,pairwise=True):
        edges = self._to_attraction(borders=False,pairwise=pairwise)
        edges = edges[(edges['len']>lthr)&(edges['sum']>sthr)]
        edges['ratio'] = edges['sum']/edges['len']
        edges = edges.loc[~((edges['len']<10)&(edges.ratio>3))]
        
        ehom = self.exon_homology(lthr,sthr,pairwise,edges.copy())
        ihom = self.intron_homology(lthr,sthr,pairwise,edges.copy())
        
        ih = ihom.groupby(['ihomogroup',
                           'seqname']).inum.agg([min,max]).apply(lambda x: tuple(range(x[0]+1,x[1]+1)),1)
        candidates = []
        for _,row in ih[ih!=()].reset_index().iterrows():
            for num in row[0]:
                r = row.copy()
                r[0] = num
                candidates.append(r)
    
        if not candidates:
            return None,ehom,ihom
        ih = pd.concat(candidates,1).transpose().rename(columns={0:'exon_group'})
        
        m = ih.merge(ehom[['seqname','exon_group','homogroup']],how='left')
        seqs = set(self.dict.keys())
        seqg = {key:self.dict[key].genome for key in seqs}
        m = m.groupby(['ihomogroup','homogroup']).seqname.agg(set).apply(lambda x: seqs-x).reset_index()
    
        lst = m.homogroup.apply(lambda x: self.get_ehom_fasta(ehom,x)).tolist()
        m[['msa','enum']] = pd.DataFrame(lst,index=m.index)
        #m['enum'] = m.homogroup.apply()
        m['msa_size'] = len(self.dict)-m.seqname.apply(len)
        
        candidates = []
        for _,row in m.iterrows():
            for num in row.seqname:
                r = row.copy()
                r.seqname = num
                candidates.append(r)
        if not candidates:
            return None,ehom,ihom
        candidates = pd.concat(candidates,1).transpose()
        candidates['genome'] = candidates.seqname.map(seqg)
        
        ih = ihom.groupby(['seqname','ihomogroup','chrn','strand']).agg({'start':min,'end':max,'inum':[min,max]}).reset_index()
        inum = ih.inum.apply(lambda x: str(x[0]) if x[0]==x[1] else f"{x[0]}-{x[1]}",1)
        ih = ih.drop('inum',1)
        ih['inum'] = ih.seqname+"_"+inum
        ih.columns = [i[0] for i in ih.columns]
    
        m = candidates.merge(ih)
        def func(row):
            name = f">{row.genome}.{row.seqname}.{row.chrn}.{row.start}.{row.end}.{row.strand}"
            fa = []
            #fa.append(f">{row.genome}.{row.seqname}.{row.chrn}.{row.start}.{row.end}.{row.strand}")
            nt = Sequence.GENOMES[row.genome][row.chrn][row.start:row.end]
            if row.strand=='-':
                nt = Seq(nt).reverse_complement()
            for i in range(3):
                fa.append(name+f'.{i}')
                fa.append(str(nt[i:].translate()))
            return '\n'.join(fa)
        m['fasta'] = m.apply(func,1)
        m = m.groupby(['homogroup','msa','enum','msa_size'])[['fasta','inum']].agg(list).reset_index()
        m['dbsize'] = m.fasta.apply(len)
        m.fasta = m.fasta.apply(lambda x: '\n'.join(x))
        m.inum = m.inum.apply(lambda x: ', '.join(x))
        return m,ehom,ihom
    
    def get_ehom_fasta(self,ehom,h,frac=0.3):
        imin = np.inf
        imax = -np.inf
        msa = self.msa
        bord = self._msa2bordernum(borders=False)
        df = ehom[ehom.homogroup==h]
        func = lambda x: str(x[0]) if x[0]==x[1] else f"{x[0]}-{x[1]}"
        exons = df.groupby('seqname').exon_group.agg([min,max]).apply(func,1)
        exons = ', '.join(exons.index+"_"+exons)
        seqs = df.seqname.unique()
        for _,row in df.iterrows():
            tmp = bord[bord[row.seqname]==row.exon_group].index
            imin = min(tmp[0],imin)
            imax = max(tmp[-1],imax)
        msa = msa.loc[imin:imax,seqs]
        msa = msa[~(msa=='-').apply(all,1)]
    
        nseq = max(int(np.round(msa.shape[1]*frac)),1)
        condition = (msa!='-').apply(sum,1)>=nseq
        msa = msa.loc[condition].apply(lambda x: "".join(x))
        fasta = []
        for key,seq in msa.items():
            fasta.append('>'+key)
            fasta.append(seq)
        return '\n'.join(fasta),exons
        
    
    
class Border():
    def __init__(self,dist,l,seq,exon,num,repulsion={},attraction=dict()):
        self.dist = dist #position of border minus position of parent exon
        self.l = l #length of border (needed for repulsion)
        self.seq = seq #parent sequence
        self.exon = exon #parent exon
        self.num = num #num of border
        self.repulsion = repulsion #set of border neighbors' indices (||||---||||)
        self.attraction = attraction #dict of border attractors (seq,num):score
    def __repr__(self):
        return str(self.__dict__)
        
class Node():
    def __init__(self,x,l,seq,num,borders):
        self.x = x #position of exon
        self.l = l #length of exon
        self.seq = seq #parent sequence
        self.num = num #num of exon
        self.borders = borders #list of border indices forming exon
    def __repr__(self):
        return str(self.__dict__)

class Graph():
    def __init__(self,nodes,borders,k,scorethr=-np.inf,neighbors=None):
        self.nodes = nodes
        self.borders = borders
        self.k = k
        self.scorethr = scorethr
        if neighbors is None:
            ne = [x for xs in self.borders.values() for x in xs.attraction.values()]
            neighbors = 0.7 * np.median(ne)
        self.neighbors = neighbors
        

    def get_dx(self,b1,b2):
        x1 = self.borders[b1]
        x2 = self.borders[b2]
        x1 = x1.dist + self.nodes[(x1.seq,x1.exon)].x
        x2 = x2.dist + self.nodes[(x2.seq,x2.exon)].x
        l = abs(self.borders[b1].l+self.borders[b2].l)/2
        delta = x1 - x2
        sgn = 0 if delta==0 else delta/abs(delta)
        delta = abs(delta) - l
        if delta<0:
            delta = 1
        if np.isnan(delta):
            print(b1,b2,'delta')
        return abs(delta),sgn   
    
    def attraction(self,b1,b2):
        if self.borders[b1].seq==self.borders[b2].seq:
            return 0
        score = self.borders[b1].attraction[b2]
        dx,sgn = self.get_dx(b1,b2)
        #print(dx,sgn,score)
        return -sgn*score*dx/self.k
    
    def repulsion(self,b1,b2):
        dx,sgn = self.get_dx(b1,b2)
        delta = self.k - abs(dx)
        sgn2 = 0 if delta==0 else delta/abs(delta)
        val = sgn*sgn2*delta**2/self.k
        return val
    
    def move(self,t,repcoef,debug=False):
        for (seq,num),n in self.nodes.items():
            att = 0
            rep = 0
            for b1 in n.borders:
                for b2 in self.borders[b1].attraction.keys():
                    att+=self.attraction(b1,b2)
                for b2 in self.borders[b1].repulsion:
                    rep+=self.repulsion(b1,b2)
            #print(f"Iteration={t}, node=({seq},{num}), repulsion={rep}, attraction={att}")
            F = t*(rep*repcoef+att)
    #calculate borders
            if num==1:
                lb = 0
            else:
                lnode = self.nodes[(seq,num-1)]
                lb = lnode.x+lnode.l/2+50+n.l/2
            rkey = (seq,num+1)
            if rkey in self.nodes.keys():
                rnode = self.nodes[rkey]
                rb = rnode.x-rnode.l/2-50-n.l/2
            else:
                rb = lnode.x+10000
            moved = n.x+F
            pos = min(max(moved,lb),rb)
    #            if F>self.k:
    #                print('move',n2,F,pos-self.nodes[n2].x)
            if debug:
                print('move',(seq,num),F,pos-self.nodes[(seq,num)].x)
            self.nodes[(seq,num)].x = pos
    
    @staticmethod
    def exp_t(t,iterations,a):
        ymin = t/iterations
        xmin = (-(np.log(ymin/t)))**(1/a)
        x = np.linspace(0,xmin,iterations)
        y = t*np.exp(-(x**a))
        return y
    
    def FD(self,iterations=50,t=None,repcoef=1,a=1):
        if t is None:
            t = self.k*0.1
        tarr = Graph.exp_t(t,iterations,a)
        for t in tarr:
            self.move(t,repcoef)
    
    def to_ilength(self,minlen=20):
        df = pd.DataFrame([[*i,j.x,j.l] for i,j in self.nodes.items()],columns=['seq','num','pos','l'])
        df['left'] = df.pos-df.l/2
        df['right'] = df.pos+df.l/2
        df['prev'] = df.groupby('seq').right.transform(lambda x: [0]+x.tolist()[:-1])
        df['intron'] = df.left-df.prev
        df.loc[df.intron<minlen,'intron'] = minlen
        first = df[df.num==1]
        mini = min(first.intron)
        df.loc[first.index,'intron']-=mini
        return df.groupby('seq').intron.agg(list).to_dict()

    def plot(self):
        f,ax = plt.subplots(figsize=(10,5))
        #names = {j[0]:i for i,j in enumerate(self.nodes.keys())}
        names = sorted({i[0] for i in self.nodes.keys()})
        names = {j:i for i,j in enumerate(names)}
        for (seq,num), val in self.nodes.items():
            xl = val.x-val.l/2
            xr = val.x+val.l/2
            y = names[seq]
            ax.fill_between([xl,xr],y-0.2,y+0.2,color='grey')
            ax.text(val.x,y,num,va='center',ha='center')
        ax.set_yticks(list(names.values()),list(names.keys()))
        return f

    def to_table(self):
        #returns 
        #1 dataframe of new coordinates of borders
        #2 dataframe of new coordinates of exons and introns
        res = []
        for (seq,num),b in self.borders.items():
            e = self.nodes[(b.seq,b.exon)]
            es = e.x-e.l/2
            ee = e.x+e.l/2
            x = b.dist + e.x
            bs = x-b.l/2
            be = x+b.l/2
            res.append([seq,e.num,num,es,ee,bs,be])
        df = pd.DataFrame(res,columns=['seqname','exon_group','ind','estart','eend','bstart','bend'])
        mini = min(df.bstart)
        df[['bstart','bend','estart','eend']]-=mini
        introns = df[['seqname','exon_group','estart','eend']].drop_duplicates()
        exons = introns.copy()
        exons['type'] = 'exon'
        introns['prev'] = introns.groupby('seqname').eend.transform(lambda x: [np.nan]+x.tolist()[:-1])
        introns = introns[introns.prev.notna()].drop('eend',1).rename(columns={'prev':'estart','estart':'eend'})
        introns['type'] = 'intron'
        new = pd.concat([exons,introns])
        return df,new
    
### Convert genomic coordinates to plot coordinates

def convert_coords(lst,tab=None,fam=None,G=None):
    #converts genomic coodinates in the form [('seqname',x)] to plot coordinates
    res = np.zeros(len(lst))
    if tab is None:
        old = fam.to_table()
        new = G.to_table()[1]
        tab = old.merge(new)
        tab.loc[tab.strand=='-',['estart','eend']] = tab.loc[tab.strand=='-',['estart','eend']].rename(columns={'estart':'eend',
                                                                                                                'eend':'estart'})
    lst = pd.DataFrame(lst,columns=['seqname','x'])
    lst['id'] = lst.index
    mask = lst.seqname.isin(tab.seqname.unique())&(lst.x.notna())
    res[~mask] = np.nan
    lst = lst[mask]
    lst = lst.merge(tab,how='left')
    
    inner = lst[(lst.istart<=lst.x)&(lst.x<lst.iend)]
    inner['a'] = (inner.eend-inner.estart)/(inner.iend-inner.istart)
    inner['b'] = inner.eend-inner.a*inner.iend
    inner['newx'] = inner.x*inner.a+inner.b
    inner = inner[['id','newx']].drop_duplicates()
    res[inner.id.tolist()] = inner.newx.tolist()
    lst = lst[~lst.id.isin(inner.id.unique())]
    if len(lst!=0):
        l1 = lst.drop(['iend','eend'],1)
        l2 = lst.drop(['istart','estart'],1).rename(columns={'iend':'istart','eend':'estart'})
        df = pd.concat([l1,l2]).groupby(['strand','x','id'])[['istart','estart']].agg([min,max]).reset_index()
        df.columns = ['_'.join(i).strip('_') for i in df.columns]
        df['newx'] = df.apply(edge_coord,1)
        res[df.id.tolist()] = df.newx.tolist()
    return res

def edge_coord(row):
    if row.x>=row.istart_max:
        delta = row.x-row.istart_max
        if row.strand=='+':
            return row.estart_max+delta
        return row.estart_min-delta
    else:
        delta = row.istart_min-row.x
        if row.strand=='+':
            return row.estart_min-delta
        return row.estart_max+delta


### Search unannotated exons via blast

def search_missing(row,wd='./',famname=None,evalue=1,blastpath=None,windows=True):
    if blastpath is None:
        blastpath = r"C:\Program Files\NCBI\blast-2.17.0+\bin\\"
    if not os.path.exists(wd):
        subprocess.run(f'mkdir -p "{wd}"',shell=True)
    query = row.msa
    subject = row.fasta
    psi = True if row.msa_size>1 else False
#    if row.dbsize>1:
    db=True
    subject = makeblastdb(row,wd,psi,famname=famname,blastpath=blastpath,windows=windows)
#    else:
#        db = False
#        subject = write_fasta(row,wd,famname)
    return blast(query,subject,db,psi,evalue,blastpath=blastpath,windows=windows)

def blast(query,subject,db,psi,evalue,blastpath,windows):
    outfmt = '"6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qseq sseq"'
    sbj = '-db' if db else '-subject'
    ext = '.exe"' if windows else ''
    blast = f"psiblast{ext} -comp_based_stats 1 -in_msa" if psi else f"blastp{ext} -query"
    exe = f'{blastpath}{blast}'
    if windows:
        exe = f'"{exe}'
    command = f'{exe} - {sbj} {subject} -outfmt {outfmt} -window_size 0 -evalue {evalue}  -word_size 2'
    #print(command)
    p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,err = p.communicate(input=query.encode())
    out,err = out.decode(),err.decode()
    if p.returncode!=0:
        raise Exception(err)
    if err:
        #warnings.warn(err)
        print(err)
    with contextlib.suppress(FileNotFoundError):
        if subject.endswith('.fasta'):
            os.remove(subject)
        else:
            for ext in ['.pto', '.phr', '.pog', '.pdb', '.psq', '.phd', '.ptf', '.pin', '.phi', '.pot', '.pjs']:
                os.remove(subject+ext)
    names = 'qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qseq sseq'.split()
    res = pd.read_csv(StringIO(out),sep='\t',header=None,names=names)
    #res[['qstart','sstart']]-=1
    return res

def makeblastdb(row,wd,psi,famname,blastpath,windows):
    pid = os.getpid()
    name = f'{wd}pid{pid}homogroup{row.homogroup}'
    if famname is not None:
        name+='famname'+famname
    dbtype = 'prot'
    #dbtype = 'prot' if psi else 'nucl'
    ext = '.exe"' if windows else ''
    exe = f'{blastpath}makeblastdb{ext}'
    if windows:
        exe = f'"{exe}'
    cmd = f'{exe} -hash_index -title {name.split("/")[-1]} -in - -dbtype {dbtype} -out {name}'
    #print(cmd)
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,err = p.communicate(input=row.fasta.encode())
    out,err = out.decode(),err.decode()
    if p.returncode!=0:
        raise Exception(err)
    if err:
        #warnings.warn(err)
        print(err)
    return name

def write_fasta(row,wd,famname=None):
    pid = os.getpid()
    name = f'{wd}pid{pid}_homogroup{row.homogroup}'
    if famname is not None:
        name+='_famname'+famname
    name+='.fasta'
    with open(name,'w') as f:
        f.write(row.fasta)
    return name

def blast2genome(row):
    genome,seqname,chrn,start,end,strand,frame = row.sseqid.split('.')
    start,end,frame = int(start),int(end),int(frame)
    gstart = (row.sstart-1)*3+frame
    gend = row.send*3+frame
    if strand=='+':
        gstart+=start
        gend+=start
    else:
        gstart,gend = end-gend,end-gstart
    return [seqname,-1,chrn,strand,gstart,gend,row.homogroup,0]