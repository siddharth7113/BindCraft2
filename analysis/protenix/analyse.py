#!/usr/bin/env python3
"""Compare Protenix predictions against BC2's designs.

The question is NOT whether the confidence scores agree -- different models, different scales.
It is whether Protenix independently places each peptide on the SAME epitope BC2 designed it for.
BC2 designed with AlphaFold2 and validated with held-out AlphaFold2 models; Protenix is a
different architecture, so this is the first independent opinion these designs get.
"""
import csv, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOTSPOTS = {35, 62, 64, 67, 93, 95, 96, 98, 99}

def read_cif_atoms(path):
    hdr, inloop, rows = [], False, []
    for line in open(path):
        s = line.strip()
        if s.startswith('_atom_site.'):
            hdr.append(s.split('.')[1]); inloop = True; continue
        if inloop and s.startswith(('ATOM', 'HETATM')):
            f = s.split()
            if len(f) == len(hdr): rows.append(dict(zip(hdr, f)))
        elif inloop and s == '#':
            inloop = False
    return rows

def xyz(a): return (float(a['Cartn_x']), float(a['Cartn_y']), float(a['Cartn_z']))
def chain(a): return a.get('auth_asym_id') or a.get('label_asym_id')
def resnum(a):
    for k in ('auth_seq_id', 'label_seq_id'):
        if a.get(k, '.') not in ('.', '?'):
            try: return int(a[k])
            except ValueError: pass
    return None

def analyse(cif):
    atoms = read_cif_atoms(cif)
    by_chain = {}
    for a in atoms: by_chain.setdefault(chain(a), []).append(a)
    if len(by_chain) < 2: return None
    # binder = chain with fewest distinct residues
    bkey = min(by_chain, key=lambda c: len({resnum(a) for a in by_chain[c]}))
    binder = by_chain[bkey]
    target = [a for c, v in by_chain.items() if c != bkey for a in v]

    # interface: target residues with any atom within 4.5 A of any binder atom
    bx = [xyz(a) for a in binder]
    contacts = set()
    for a in target:
        p = xyz(a)
        for q in bx:
            if (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2 < 20.25:
                r = resnum(a)
                if r is not None: contacts.add(r)
                break

    # ring closure: N of first residue <-> C of last residue
    rn = sorted({resnum(a) for a in binder if resnum(a) is not None})
    nc = None
    if rn:
        n1 = [a for a in binder if resnum(a) == rn[0] and a['label_atom_id'] == 'N']
        cL = [a for a in binder if resnum(a) == rn[-1] and a['label_atom_id'] == 'C']
        if n1 and cL: nc = math.dist(xyz(n1[0]), xyz(cL[0]))
    return {'binder_chain': bkey, 'binder_residues': len(rn),
            'contacts': sorted(contacts), 'n_contacts': len(contacts),
            'hotspot_hits': sorted(contacts & HOTSPOTS), 'nc_distance': nc}

def bc2_contacts(ranked_csv, rank):
    for r in csv.DictReader(open(ranked_csv)):
        if int(r['rank']) == rank:
            out = set()
            for tok in r['Interface_Target_Residues'].split(','):
                tok = tok.strip()
                if tok:
                    d = ''.join(ch for ch in tok if ch.isdigit())
                    if d: out.add(int(d))
            return out, r
    return set(), None

def seed_dirs(job):
    """Each seed is one independent opinion. Protenix writes job/seed_NNN/predictions/*.cif;
    fall back to treating the whole job as a single unlabelled seed for older layouts."""
    sd = sorted(p for p in job.glob('seed_*') if p.is_dir())
    return [(p.name.split('_', 1)[1], p) for p in sd] or [('?', job)]

def best_sample(root):
    """Protenix emits `sample` structures per seed; they are near-duplicates (spread ~0.005),
    so taking the most confident one within a seed is safe. Taking it ACROSS seeds is not --
    that reports a maximum that only grows as seeds are added."""
    best, best_iptm, conf = None, -1.0, {}
    for c in sorted(root.rglob('*.cif')):
        j = c.with_name(c.name.replace('_sample_', '_summary_confidence_sample_').replace('.cif', '.json'))
        if j.exists():
            cj = json.load(open(j))
            if cj.get('iptm', -1) > best_iptm: best, best_iptm, conf = c, cj['iptm'], cj
    return best, conf

def mean(v): return sum(v)/len(v) if v else float('nan')
def sd(v):
    if len(v) < 2: return 0.0
    m = mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))

def main():
    per_seed, jobs = [], {}
    for tag, outdir, ranked, cyc in (
        ('cyc', 'analysis/protenix/out/cyclic',  'results/rbp4_cyclic/3_Ranked/!_Ranked.csv',  True),
        ('lin', 'analysis/protenix/out/linear',  'results/rbp4_peptide/3_Ranked/!_Ranked.csv', False)):
        base = ROOT/outdir
        if not base.is_dir(): continue
        for job in sorted(base.iterdir()):
            if not job.is_dir(): continue
            try: rank = int(job.name.split('_r')[1][:2])
            except Exception: continue
            bc2, row = bc2_contacts(ROOT/ranked, rank)
            for seed, sdir in seed_dirs(job):
                cif, conf = best_sample(sdir)
                if cif is None: continue
                res = analyse(cif)
                if not res: continue
                px = set(res['contacts'])
                jac = len(px & bc2)/len(px | bc2) if (px | bc2) else 0.0
                per_seed.append({'set': tag, 'rank': rank, 'job': job.name, 'seed': seed,
                    'px_iptm': round(conf.get('iptm', float('nan')), 3),
                    'px_ptm': round(conf.get('ptm', float('nan')), 3),
                    'px_plddt': round(conf.get('plddt', float('nan')), 1),
                    'px_contacts': res['n_contacts'], 'px_hotspots': len(res['hotspot_hits']),
                    'overlap': len(px & bc2), 'jaccard': round(jac, 2),
                    'nc_dist': round(res['nc_distance'], 2) if res['nc_distance'] else None,
                    'px_residues': ','.join(map(str, res['contacts']))})
                jobs.setdefault(job.name, {'set': tag, 'rank': rank, 'cyclic': cyc,
                    'bc2_iptm': row['i_pTM'] if row else '', 'seeds': [], 'contacts': []})
                jobs[job.name]['seeds'].append(per_seed[-1])
                jobs[job.name]['contacts'].append(px)
    if not per_seed:
        print("no Protenix outputs yet"); return

    per_seed.sort(key=lambda r: (r['set'], r['rank'], r['seed']))
    p = ROOT/'analysis/protenix/agreement_per_seed.csv'
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0])); w.writeheader(); w.writerows(per_seed)

    rows = []
    for name, j in jobs.items():
        ip = [s['px_iptm'] for s in j['seeds']]; ja = [s['jaccard'] for s in j['seeds']]
        nc = [s['nc_dist'] for s in j['seeds'] if s['nc_dist'] is not None]
        n = len(j['seeds'])
        # a residue is consensus if it is contacted in at least half the seeds
        tally = {}
        for c in j['contacts']:
            for r in c: tally[r] = tally.get(r, 0) + 1
        cons = sorted(r for r, k in tally.items() if k*2 >= n)
        rows.append({'set': j['set'], 'rank': j['rank'], 'job': name, 'bc2_iptm': j['bc2_iptm'],
            'cyclic': j['cyclic'], 'n_seeds': n,
            'px_iptm_mean': round(mean(ip), 3), 'px_iptm_sd': round(sd(ip), 3),
            'px_iptm_min': min(ip), 'px_iptm_max': max(ip),
            'jaccard_mean': round(mean(ja), 2), 'jaccard_sd': round(sd(ja), 2),
            'jaccard_min': min(ja), 'jaccard_max': max(ja),
            'hotspots_mean': round(mean([s['px_hotspots'] for s in j['seeds']]), 1),
            'nc_dist_mean': round(mean(nc), 2) if nc else None,
            'consensus_residues': ','.join(map(str, cons))})
    rows.sort(key=lambda r: (r['set'], r['rank']))

    print(f"{'set':>4} {'rk':>3} {'BC2':>5} {'PXipTM(sd)':>13} {'Jac(sd)':>12} {'N-C':>6} {'n':>2}  verdict")
    for r in rows:
        v = 'SAME EPITOPE' if r['jaccard_mean'] >= 0.4 else ('partial' if r['jaccard_mean'] >= 0.2 else 'DIFFERENT SITE')
        # a design whose epitope moves between seeds is not a reliable candidate, whatever its mean
        if r['n_seeds'] > 1 and r['jaccard_sd'] >= 0.15: v += '  [UNSTABLE]'
        ring = f"{r['nc_dist_mean']:.2f}" if r['nc_dist_mean'] is not None else '  -  '
        print(f"{r['set']:>4} {r['rank']:>3} {r['bc2_iptm']:>5} "
              f"{r['px_iptm_mean']:>6.3f}({r['px_iptm_sd']:.3f}) {r['jaccard_mean']:>5.2f}({r['jaccard_sd']:.2f}) "
              f"{ring:>6} {r['n_seeds']:>2}  {v}")
    out = ROOT/'analysis/protenix/agreement.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out.relative_to(ROOT)} ({len(rows)} designs) and "
          f"{p.relative_to(ROOT)} ({len(per_seed)} design-seeds)")
    cyc = [r for r in rows if r['cyclic'] and r['nc_dist_mean'] is not None]
    if cyc:
        import statistics as st
        m = st.median(r['nc_dist_mean'] for r in cyc)
        print(f"\nRING CHECK: median N-to-C distance over {len(cyc)} cyclic designs = {m:.2f} A")
        print("  ~1.3 A  -> covalent_bonds honoured, genuine macrocycles")
        print("  >5 A    -> cyclisation IGNORED; cyclic predictions are of linear analogs")

if __name__ == '__main__':
    main()
