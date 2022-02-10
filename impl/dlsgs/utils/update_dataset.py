import sys, os
import random

import sympy.logic as syl

from dlsgs.utils import ltl_parser
from dlsgs.utils.utils import nice_open


def add(d, key, item):
    if key in d:
        d[key].append(item)
    else:
        d[key] = [item]


def run(infile, target, outfile=None):
    target_, *params_ = target.split('=')
    write_through = target_ in ['relaxed_sat', 'unique', 'step_to', 'limit_length']
    with nice_open(infile, 'r') as inf, nice_open(outfile, 'w') as outf:
        sats, unsats, whatever = {}, {}, []
        formulas = set()
        for formula_line in inf:
            if formula_line.startswith('#'):
                if formula_line.startswith('#step ') and target_ == 'step_to':
                    step = int(formula_line[6:].strip())
                    up_to = int(params_[0])
                    if step > up_to:
                        break
                if write_through:
                    outf.write(formula_line)
                continue

            formula_line = formula_line.strip()
            trace_line = next(inf).strip()
            formula_line_parts = [q.strip() for q in formula_line.split('#')]
            formula = formula_line_parts[0]
            try:
                formula_obj = ltl_parser.ltl_formula(formula, 'network-polish')
            except ltl_parser.ParseError as e:
                print(e)
                print(formula_line)
                print(trace_line)
                continue

            if target == 'relaxed_sat':
                formula_nnf = formula_obj.negation_normal_form()
                relaxed = formula_nnf.relax_to_prop()
                relaxed_sympy = relaxed.to_sympy()
                #relaxed_sympy_cnf = syl.boolalg.to_cnf(relaxed_sympy, simplify=True)
                sat = syl.inference.satisfiable(relaxed_sympy, algorithm='pycosat')
                formula_line_parts.append('relaxed_sat: ' + ('1' if sat else '0'))
                if not sat and trace_line != '0':
                    raise ValueError('Very bad!')
                assert formula_obj.to_str('network-polish') == formula
                
            elif target == 'balance_per_size':
                size = formula_obj.size()
                if trace_line != '0':
                    add(sats, size, (formula_line, trace_line))
                else:
                    add(unsats, size, (formula_line, trace_line))
            
            elif target.startswith('shuffle+split'):
                whatever.append((formula_line, trace_line))

            elif target == 'unique':
                if formula in formulas:
                    continue
                formulas.add(formula)

            elif target_ == 'limit_length':
                if formula_obj.size() > int(params_[0]):
                    continue

            if write_through:
                outf.write(' #'.join(formula_line_parts) + '\n')
                outf.write(trace_line + '\n')


        
        if target == 'balance_per_size':
            sizes = sorted(set(sats) | set(unsats))
            for size in sizes:
                these_sats = sats[size] if size in sats else []
                these_unsats = unsats[size] if size in unsats else []
                goal = min(len(these_sats), len(these_unsats))
                random.shuffle(these_sats)
                random.shuffle(these_unsats)
                for fl, tl in these_sats[:goal] + these_unsats[:goal]:
                    outf.write(fl + '\n' + tl + '\n')
        
        elif target.startswith('shuffle+split'):
            random.shuffle(whatever)
            splitinfo = { k : float(v)  for k, v in [qq.split(':') for qq in target.split('=')[1].split(',')] }
            print('splitting as', splitinfo)
            total_val = sum(splitinfo.values())
            current_val = 0
            num_total = len(whatever)
            for split, val in splitinfo.items():
                items = whatever[int(current_val/total_val * num_total) : int((current_val + val)/total_val * num_total)]
                current_val += val
                path = os.path.join(outf, split + '.txt') if outf else split + '.txt'
                with open(path, 'w') as splitf:
                    for fl, tl in items:
                        splitf.write(fl + '\n' + tl + '\n')



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:', sys.argv[0], 'INFILE', 'relaxed_sat|balance_per_size|shuffle+split=X:Y,..|unique|step_to=N|limit_length=N', '[OUTFILE]')
        sys.exit(1)
    run(*sys.argv[1:])
