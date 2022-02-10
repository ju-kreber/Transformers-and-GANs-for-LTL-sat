import math, operator
import sys
import matplotlib.pyplot as plt
from dlsgs.utils import ltl_parser


"""Analyzes lengths of formulas,witnesses,sat or numerical values of dataset"""

def increment(d, l, by=1):
    if l in d:
        d[l] += by
    else:
        d[l] = by


def analyze(file, targets):
    targets = [q.strip() for q in targets.split(',')]
    data = {'formula_lengths' : {}}
    FL = data['formula_lengths']
    val_name = None
    for t in targets:
        if t.startswith('witness'):
            data['trace_lengths'] = {}
        elif t.startswith('sat'):
            data['sats'] = {}
            data['unsats'] = {}
            if 'relaxed' in t:
                data['relaxed_unsats'] = {}
        elif t.startswith('val'):
            data['val'] = {}
            val_name = t.split('=')[1]

    with open(file) as f:
        for formula_line in f:
            if formula_line.startswith('#'):
                continue
            trace_line = next(f).strip()
            formula_line = [q.strip() for q in formula_line.split('#')]
            formula = formula_line[0]
            try:
                formula_size = ltl_parser.ltl_formula(formula, 'network-polish').size()
                increment(data['formula_lengths'], formula_size)
            except ltl_parser.ParseError as e:
                print(e)
                print(formula_line)
                print(trace_line)
                continue
            if val_name is not None:
                for q in formula_line[1:]:
                    k, v = q.split(': ')
                    if k == val_name:
                        increment(data['val'], formula_size, float(v))
            #formula_lengths[0] = 1 # todo needed?
            if 'sats' in data:
                if 'relaxed_unsats' in data:
                    relaxed_sat = [v for k, v in [q.split(': ') for q in formula_line[1:]] if k == 'relaxed_sat']
                    relaxed_sat = relaxed_sat[0] if len(relaxed_sat) > 0 else None
                    if relaxed_sat is not None and relaxed_sat == '0':
                        increment(data['relaxed_unsats'], formula_size)
                if trace_line == '0':
                    increment(data['unsats'], formula_size)
                elif len(trace_line) > 0:
                    increment(data['sats'], formula_size)
                
            if 'trace_lengths' in data:
                trace_size = len(trace_line) # todo parse?
                increment(data['trace_lengths'], trace_size)

    FMIN = 1
    fmin, fmax = max(min(FL), FMIN), max(FL)
    fx = list(range(fmin, fmax+1))
    fy = [FL[fq] if fq in FL else 0 for fq in fx]

    num_subplots = len(targets)
    axes = [plt.subplots(1, 1, figsize=[11, 4])[1] for _ in range(num_subplots)]
    for i, target in enumerate(targets):
        axis = axes[i]
        if target.startswith('formula'):
            axis.bar(fx, fy, edgecolor='white', width=1, color='#4d85ff')
            # axis.set_yscale('log')
            axis.set_xlabel('formula length')
            axis.set_ylabel('number of items')
        elif target.startswith('witness'):
            TMAX = 1000
            TL = data['trace_lengths']
            tmin, tmax = min(TL), min(max(TL), TMAX)
            tx = list(range(tmin, tmax+1))
            ty = [TL[txq] if txq in TL else 0 for txq in tx]
            axis.bar(tx, ty, edgecolor='white', width=1, color='#4d85ff')
            # axis.set_yscale('log')
            axis.set_xlabel('trace length')
            axis.set_ylabel('number of items')
        elif target.startswith('sat'):
            sat_y = [data['sats'][fxq] / fyq if fxq in data['sats'] else 0 for fxq, fyq in zip(fx, fy)]
            axis.bar(fx, sat_y, edgecolor='white', width=1, color='#0eb7b1', label='sat')
            if not 'relaxed_unsats' in data:
                unsat_y = [data['unsats'][fxq] / fyq if fxq in data['unsats'] else 0 for fxq, fyq in zip(fx, fy)]
                axis.bar(fx, unsat_y, edgecolor='white', width=1, color='#ffe156', label='unsat', bottom=sat_y)    
            else:
                unsat_nontrivially_y = [(data['unsats'][fxq] - data['relaxed_unsats'][fxq]) / fyq if fxq in data['relaxed_unsats'] else (
                    data['unsats'][fxq] / fyq if fxq in data['unsats'] else 0) for fxq, fyq in zip(fx, fy)]
                axis.bar(fx, unsat_nontrivially_y, edgecolor='white', width=1, color='#ffe156', label='unsat temp. only', bottom=sat_y)
                bottom = [x + y for x,y in zip(sat_y, unsat_nontrivially_y)]
                unsat_trivially_y = [data['relaxed_unsats'][fxq] / fyq if fxq in data['relaxed_unsats'] else 0 for fxq, fyq in zip(fx, fy)]
                axis.bar(fx, unsat_trivially_y, edgecolor='white', width=1, color='#ffa756', label='unsat non-temp.', bottom=bottom)
            axis.set_xlabel('formula length')
            axis.set_ylabel('percentage')
            handles, labels = axis.get_legend_handles_labels() # thanks https://stackoverflow.com/questions/34576059/reverse-the-order-of-legend
            axis.legend(handles[::-1], labels[::-1], loc='lower left')
        elif target.startswith('val'):
            val_y = [data['val'][fxq] / fyq if fxq in data['val'] else 0 for fxq, fyq in zip(fx, fy)]
            axis.bar(fx, val_y, edgecolor='white', width=1, color='#4d85ff')
            axis.set_xlabel('formula length')
            axis.set_ylabel('average ' + val_name)
        if target.endswith('^'):
            axis.set_yscale('log')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], 'FILE', '[formula|witness|sat[+relaxed]|valavg=xyz comma separated, optinal ^ at end for log]')
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else 'formula,witness')
