
# pylint: disable=line-too-long

from functools import reduce
import os, re, subprocess
from timeit import default_timer as timer
import random
from math import log

import sympy.logic as syl
#syl = importlib.import_module('sympy.logic') # workaround vscode
from sympy.assumptions.cnf import EncodedCNF

from dlsgs.utils import ltl_parser

DEFAULT_BINARY_PATH = 'bin'


def to_dimacs(formula, sample_set=None):
    # taken from sympy's pycosat_wrapper
    if not isinstance(formula, EncodedCNF):
        cnf = EncodedCNF()
        cnf.add_prop(formula)
    assert not {0} in cnf.data
    num_clauses = len(cnf.data)
    all_literals = reduce(lambda a, b: a | b, cnf.data, set())
    all_variables = map(abs, all_literals)
    highest_variable = max(all_variables)
    res = f'c generated formula\np cnf {highest_variable:d} {num_clauses:d}\n'
    if sample_set:
        res += sample_set
    res += ' 0\n'.join([' '.join(map(str, clause)) for clause in cnf.data]) + ' 0\n'
    return res


def approximate_model_count(formula, binary_path=DEFAULT_BINARY_PATH):
    formula_dimacs = to_dimacs(formula, sample_set=None)
    res = subprocess.run([os.path.join(binary_path, 'approxmc'), '-v0'], input=formula_dimacs, text=True, capture_output=True)
    if res.returncode != 0:
        raise ValueError('mc tool failed with returncode ' + str(res.returncode) + ', stderr:\n' + res.stderr)
    m = re.search(r's mc (\d+)$', res.stdout)
    if not m:
        raise ValueError('Could not find mc in mc tool output')
    mc = int(m.groups()[0])
    return mc


def solve_prop(formula_obj, tool, solution_choice, simplify=True, count_models=False, model_counting='naive', binary_path=DEFAULT_BINARY_PATH):
    assert tool == 'sympy'
    d = {}
    formula_sym = formula_obj.to_sympy()
    d['model_poss'] = 2**len(formula_sym.atoms())
    d['log_model_poss'] = len(formula_sym.atoms())
    if simplify:
        formula_cnf = syl.boolalg.simplify_logic(formula_sym, form='cnf')
    else:
        formula_cnf = syl.boolalg.to_cnf(formula_sym, simplify=False)
    if simplify:
        d['simplified_formula'] = ltl_parser.from_sympy(formula_cnf)
    all_models = solution_choice in ['all', 'random'] or (count_models and model_counting == 'naive')
    t_start = timer()
    res = syl.inference.satisfiable(formula_cnf, algorithm='pycosat', all_models=all_models)
    d['solve_time'] = (timer() - t_start) * 1000
    if all_models:
        if model_counting != 'naive':
            print('WARNING: model_counting not set to naive, but all models should be computed. This does not make sense')
        models = list(res)
        if not models[0]: # unsat
            d['model_count'] = 0
            return False, None, d
        else: # sat
            removed_variables = formula_sym.atoms() - formula_cnf.atoms()
            d['model_count'] = len(models) * 2** len(removed_variables)
            d['log_model_count'] = log(len(models), 2) + len(removed_variables)
            d['log_model_frac'] = d['log_model_count'] / d['log_model_poss']
            d['model_frac'] = d['model_count'] / d['model_poss']
            if solution_choice == 'all':
                return True, models, d
            if solution_choice == 'random':
                random.shuffle(models)
            if solution_choice in ['random', 'first']:
                return True, models[0], d
            else:
                raise ValueError()
    else: # only one model
        assert solution_choice == 'first' or solution_choice is None
        if count_models and res:
            assert model_counting == 'approximate'
            model_count = approximate_model_count(formula_sym, binary_path=binary_path)
            d['model_count'] = model_count
            d['model_frac'] = d['model_count'] / d['model_poss']
            d['log_model_count'] = log(model_count, 2)
            d['log_model_frac'] = d['log_model_count'] / d['log_model_poss']
        return bool(res), res or None, d
