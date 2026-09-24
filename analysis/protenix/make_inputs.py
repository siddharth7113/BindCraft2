#!/usr/bin/env python3
"""Build Protenix inference inputs for BC2 designs against RBP4.

Cyclic designs get a head-to-tail amide bond via `covalent_bonds` -- the thing
AlphaFold Server cannot express. Entity indices are 1-based into `sequences`,
so the target is entity "1" and the binder entity "2".
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = (ROOT/'targets/RBP4.fasta').read_text().split('\n',1)[1].replace('\n','')

def jobs_from(ranked_csv, tag, cyclic, skip_ranks=()):
    out=[]
    for r in csv.DictReader(open(ranked_csv)):
        rank=int(r['rank'])
        if rank in skip_ranks: continue
        seq=r['Binder_Sequence']; L=len(seq)
        job={"name": f"{tag}_r{rank:02d}_l{L}",
             "sequences":[{"proteinChain":{"sequence":TARGET,"count":1}},
                          {"proteinChain":{"sequence":seq,   "count":1}}]}
        if cyclic:
            # head-to-tail amide: N of residue 1 <-> C of residue L, both on entity 2
            job["covalent_bonds"]=[{"entity1":"2","copy1":1,"position1":1,"atom1":"N",
                                    "entity2":"2","copy2":1,"position2":L,"atom2":"C"}]
        out.append(job)
    return out

if __name__ == '__main__':
    specs=[('results/rbp4_cyclic/3_Ranked/!_Ranked.csv','rbp4cyc',True ,()),
           ('results/rbp4_peptide/3_Ranked/!_Ranked.csv','rbp4lin',False,(5,))]  # r5 binds off-epitope
    allj=[]
    for rel,tag,cyc,skip in specs:
        js=jobs_from(ROOT/rel, tag, cyc, skip)
        p=ROOT/f'analysis/protenix/{tag}_input.json'
        json.dump(js, open(p,'w'), indent=2)
        print(f"{p.relative_to(ROOT)}: {len(js)} jobs, cyclic={cyc}" + (f", skipped rank {skip}" if skip else ""))
        allj+=js
    p=ROOT/'analysis/protenix/all_input.json'
    json.dump(allj, open(p,'w'), indent=2)
    print(f"{p.relative_to(ROOT)}: {len(allj)} jobs total")
