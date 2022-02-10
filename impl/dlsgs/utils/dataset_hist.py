import math, operator
import sys
import matplotlib.pyplot as plt
from dlsgs.utils import ltl_parser


def run(file, target, bins):
    vals = {}
    vals_sat = {}
    vals_unsat = {}
    data = { k : [] for k in ['trace_steps', 'trace_step_variables', 'formula_size'] }
    sat = 0
    unsat = 0
    total = 0


    with open(file) as f:
        for formula_line in f:
            total += 1
            trace_line = next(f).strip()
            formula_line = [q.strip() for q in formula_line.split('#')]
            formula = formula_line[0]
            try:
                formula_size = ltl_parser.ltl_formula(formula, 'network-polish').size()
            except ltl_parser.ParseError as e:
                print(e)
                print(formula_line)
                print(trace_line)
                continue
            data['formula_size'].append(formula_size)
            for q in formula_line[1:]:
                k, v = q.split(': ')
                try:
                    v = float(v)
                    for d in vals, (vals_unsat if trace_line == '0' else vals_sat):
                        if k in d:
                            d[k].append(v)
                        else:
                            d[k] = [v]
                except ValueError:
                    pass
            
            if trace_line == '0':
                unsat += 1
            else:
                try:
                    trace = ltl_parser.ltl_trace(trace_line, 'network-polish')
                except ltl_parser.ParseError as e:
                    print(e)
                    print(trace_line)
                    continue
                trace_steps = len(trace.prefix_formulas) + len(trace.cycle_formulas)
                total_aps = 0
                for step in trace.prefix_formulas + trace.cycle_formulas:
                    total_aps += len(step.contained_aps())
                data['trace_steps'].append(trace_steps)
                data['trace_step_variables'].append(total_aps / trace_steps)
                sat += 1

    print('Main averages:')
    print('\tsat:   ', f'{sat / total * 100:.1f}%')
    print('\tunsat: ', f'{unsat / total * 100:.1f}%')
    print('\ttotal: ', total, 'examples')
    for k, v in data.items():
        print('\t' + k + ':', f'{sum(v) / len(v):.2f}')
    print('Tag averages:')
    for k, v in vals.items():
        print('\t' + k + ':', f'{sum(v) / len(v):.2f}')
    print('Tag averages sat:')
    for k, v in vals_sat.items():
        print('\t' + k + ':', f'{sum(v) / len(v):.2f}')
    print('Tag averages unsat:')
    for k, v in vals_unsat.items():
        print('\t' + k + ':', f'{sum(v) / len(v):.2f}')

    if target is not None:
        log = target[-1] == '^'
        if log:
            target = target[:-1]
        if target in data:
            d = data[target]
        elif target.startswith('val='):
            d = vals[target.split('=')[1]]
        else:
            raise ValueError('unknown' + target)
        plt.hist(d, bins=bins, log=log)
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], 'FILE', '[trace_steps|trace_step_variables|val=xyz] [BINS]')
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else None, int(sys.argv[3]) if len(sys.argv) >= 4 else 15)
