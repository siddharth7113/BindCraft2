#!/usr/bin/env python3
"""Build precomputed MSAs so Protenix never calls the MSA server.

Why: every job otherwise submits TWO searches -- the RBP4 target (identical across all 20 jobs)
and the binder. The binders are de novo sequences with no homologs by construction, so their
searches return nothing useful yet still sit in the remote queue (observed: 10+ min each).

Protenix skips the search when pairedMsaPath/unpairedMsaPath already exist
(runner/msa_search.py:41-56). Format, from colab_request_utils.py:300 --
  non_pairing.a3m : ">query\\n<seq>\\n" followed by homologs
  pairing.a3m     : same shape; used for cross-chain species pairing

Pairing is genuinely empty here: a de novo peptide has no homologs to pair against, so a
singleton pairing.a3m is the correct result, not a degradation.
"""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'analysis/protenix/msa_cache'
TARGET = (ROOT/'targets/RBP4.fasta').read_text().split('\n',1)[1].replace('\n','')

def write_pair(dirpath, query_seq, homolog_lines=()):
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath/'non_pairing.a3m','w') as f:
        f.write(f">query\n{query_seq}\n")
        for h, s in homolog_lines: f.write(f">{h}\n{s}\n")
    with open(dirpath/'pairing.a3m','w') as f:
        f.write(f">query\n{query_seq}\n")
    return dirpath

def parse_a3m(path):
    out, name, buf = [], None, []
    for line in open(path):
        line = line.rstrip('\n').replace('\x00','')
        if line.startswith('>'):
            if name is not None: out.append((name, ''.join(buf)))
            name, buf = line[1:], []
        elif line:
            buf.append(line)
    if name is not None: out.append((name, ''.join(buf)))
    return out

# --- target: reuse the 673-sequence alignment ColabFold returned ---
raw = parse_a3m(CACHE/'target_raw.a3m')
homologs = [(n,s) for n,s in raw if not n.startswith('query')]
tgt_dir = write_pair(CACHE/'target', TARGET, homologs)
print(f"target : {tgt_dir.relative_to(ROOT)}  ({len(homologs)} homologs + query)")

# --- binders: singletons (no homologs exist) ---
specs = [('results/rbp4_cyclic/3_Ranked/!_Ranked.csv','rbp4cyc',True,()),
         ('results/rbp4_peptide/3_Ranked/!_Ranked.csv','rbp4lin',False,(5,))]
jobs_by_tag = {}
for rel, tag, cyclic, skip in specs:
    jobs = []
    for r in csv.DictReader(open(ROOT/rel)):
        rank = int(r['rank'])
        if rank in skip: continue
        seq = r['Binder_Sequence']; L = len(seq)
        name = f"{tag}_r{rank:02d}_l{L}"
        bdir = write_pair(CACHE/'binders'/name, seq)
        job = {"name": name,
               "sequences":[
                   {"proteinChain":{"sequence":TARGET,"count":1,
                                    "pairedMsaPath":str(tgt_dir/'pairing.a3m'),
                                    "unpairedMsaPath":str(tgt_dir/'non_pairing.a3m')}},
                   {"proteinChain":{"sequence":seq,"count":1,
                                    "pairedMsaPath":str(bdir/'pairing.a3m'),
                                    "unpairedMsaPath":str(bdir/'non_pairing.a3m')}}]}
        if cyclic:
            job["covalent_bonds"]=[{"entity1":"2","copy1":1,"position1":1,"atom1":"N",
                                    "entity2":"2","copy2":1,"position2":L,"atom2":"C"}]
        jobs.append(job)
    out = ROOT/f'analysis/protenix/{tag}_input.json'
    json.dump(jobs, open(out,'w'), indent=2)
    jobs_by_tag[tag] = jobs
    print(f"{tag}   : {len(jobs)} jobs -> {out.relative_to(ROOT)}")
json.dump(sum(jobs_by_tag.values(), []), open(ROOT/'analysis/protenix/all_input.json','w'), indent=2)
print("binder MSAs: singleton (de novo -> no homologs exist)")
