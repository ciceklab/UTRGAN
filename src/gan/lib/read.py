from Bio import SeqIO
import pandas as pd
import numpy as np
# from BCBio import GFF
import pandas as pd
import requests, sys


def fetch_seq(start, end, chr, strand):
    server = "https://rest.ensembl.org"
    
    ext = "/sequence/region/human/" + str(chr) + ":" + str(start) + ".." + str(end) + ":" + str(strand) +  "?"
    
    r = requests.get(server+ext, headers={ "Content-Type" : "text/plain"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()

    return r.text

def parse_biomart(path = 'martquery_0721120207_840.txt'):

    file = path

    fasta_sequences = SeqIO.parse(open(file),'fasta')

    genes = []
    ustarts = []
    uends = []
    seqs = []
    strands = []
    tsss = []
    chromosomes = []

    counter = 0

    for fasta in fasta_sequences:

        name, sequence = fasta.id, str(fasta.seq)
        
        if sequence != "Sequenceunavailable":

            counter += 1

            listed = name.split('|')
            # print(len(listed))

            if len(listed) == 8:

                genes.append(listed[1])
                chromosomes.append(listed[2])
                
                if ';' in listed[3]:
                    ustart = listed[3].split(';')[0]
                    uend = listed[4].split(';')[0]
                else:
                    ustart = listed[3]
                    uend = listed[4]

                strand = int(listed[0])

                if strand == -1:
                    ustarts.append(ustart)
                    uends.append(uend)
                else:
                    ustarts.append(uend)
                    uends.append(ustart)


                strands.append(str(strand))
                
                tsss.append(listed[-1])
                
                seqs.append(sequence)

            # break
    # print(len(seqs))
    df = pd.DataFrame({'utr':seqs,'gene':genes, 'chr': chromosomes,'utr_start':ustarts,'utr_end':uends,'tss':tsss,'strand':strands})
    return df
