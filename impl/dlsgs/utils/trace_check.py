#!python3.6

# implementation based on DeepLTL https://github.com/reactive-systems/deepltl

# pylint: disable=line-too-long

import sys
import argparse
import math
from functools import reduce
import pickle

try:
    import spot
except ModuleNotFoundError:
    print('Warning: spot Python module not found. Use alternative checker!')

from dlsgs.utils import ltl_parser
from dlsgs.data_generation.ltl import aalta_sat
from dlsgs.utils.utils import TicToc, PersistentWorker, nice_open
from dlsgs.utils.ltl_parser import LTLTrace, LTLFormula, Token, F_AND, F_IMPLIES, F_NEXT, F_GLOBALLY, F_NOT, F_AP




def per_size_analysis(results, **kwargs):
    import matplotlib.pyplot as plt

    colors = {'unsat correct' : '#00a7ff', 'syntactically correct' : '#38b547', 'only semantically correct' : '#85f67c', 'incorrect' : '#ed974d', 'invalid' : '#fd4a4a', 'unknown' : '#a7a7a7'}
    names = {'unsat correct' : 'unsat correct', 'syntactically correct' : 'syntactic accuracy', 'only semantically correct' : 'semantic accuracy', 'incorrect' : 'incorrect', 'invalid' : 'invalid', 'unknown' : '-----'}

    if not 'load_from' in kwargs:
        min_size = min([min(d) if len(d) > 0 else math.inf for d in results.values()])
        max_size = max([max(d) if len(d) > 0 else 0 for d in results.values()])
        x, totals = [], []
        assert not ('total' in results)
        results_complete = {}
        for size in range(min_size, max_size+1):
            x.append(size)
            totals.append(0)
        bottom_positions = totals.copy()

        for category, dist in results.items(): # dict with sizes to list; not all values may occur in dict
            results_complete[category] = []
            for idx, size in enumerate(range(min_size, max_size+1)):
                value = dist[size] if size in dist else 0
                results_complete[category].append(value)
                totals[idx] += value
        results_percent = {}
        for category, dist_complete in results_complete.items():
            results_percent[category] = []
            for val, total in zip(dist_complete, totals):
                if total == 0 and val != 0:
                    raise RuntimeError()
                results_percent[category].append(val / total * 100 if total > 0 else 0)
    else: # load from file
        with open(kwargs['load_from'], 'rb') as pers_file:
            persistence = pickle.load(pers_file)
            x = persistence['x']
            results_percent = persistence['results_percent']
            bottom_positions = persistence['bottom_positions']
            totals = persistence['totals']
            results_complete = persistence['results_complete']
            results = {}

    # Do the plotting
    # thanks to https://chrisalbon.com/python/data_visualization/matplotlib_percentage_stacked_bar_plot/
    # figure, (hist_ax, dist_ax) = plt.subplots(2, figsize=(12,8))
    # figure, (hist_ax) = plt.subplots(1, figsize=(12,5))
    figure, (dist_ax) = plt.subplots(1, figsize=(12,5))
    bar_width = 1
    # hist_ax.bar(x, totals, width=bar_width, color='#3071ff', edgecolor='white')
    # hist_ax.set_ylabel('number of items')
    # hist_ax.set_xlabel('formula size')
    persistence = {'x' : x, 'results_percent' : results_percent, 'bottom_positions':bottom_positions, 'totals' : totals, 'results_complete' : results_complete}
    with open('trace_check_data.pkl', 'wb') as pers_file:
        pickle.dump(persistence, pers_file, pickle.HIGHEST_PROTOCOL)
    for category, dist_percent in results_percent.items():
        if category == 'unknown':
            continue
        dist_ax.bar(x, dist_percent, bottom=bottom_positions, label=names[category], width=bar_width, color=colors[category], edgecolor='white')
        bottom_positions = [acc + q for acc, q in zip(bottom_positions, dist_percent)] # update positions
    dist_ax.set_ylabel('percentage')
    dist_ax.set_xlabel('formula size')
    dist_ax.set_ylim(-10, 110)
    dist_ax.legend(loc='lower left')
    if 'save_analysis' in kwargs and kwargs['save_analysis'] is not None:
        figure.savefig(kwargs['save_analysis'] + '.png')
        figure.savefig(kwargs['save_analysis'] + '.svg')
    # plt.show()

    # collapse size-wise data for further processing
    results_collapsed = {}
    for category, dist in results.items():
        results_collapsed[category] = sum(dist.values())
    return results_collapsed


def encode_for_satisfiability(trace_obj: LTLTrace, formula: LTLFormula):
    # prefix
    step_constraints = []
    for idx, trace_step_formula in enumerate(trace_obj.prefix_formulas):
        for _ in range(idx): # prepend X's for step
            trace_step_formula = F_NEXT(trace_step_formula)
        step_constraints.append(trace_step_formula)
    prefix_part = reduce(F_AND, step_constraints) if step_constraints else None # AND together

    # generate encoding aps for cycle steps
    cycle_encoding_bits = bin(len(trace_obj.cycle_formulas))[2:]
    used_aps = trace_obj.contained_aps() | formula.contained_aps() # TODO: remove?
    num_encoding_aps = len(cycle_encoding_bits)
    encoding_aps = ['c' + str(q) for q in range(num_encoding_aps)]

    # build encodings for cycle steps
    encodings = []
    for idx, _ in enumerate(trace_obj.cycle_formulas):
        bin_rep = '{{:0{:d}b}}'.format(num_encoding_aps).format(idx)
        encoding = []
        for idx_encode, c in enumerate(bin_rep):
            ap = F_AP(encoding_aps[idx_encode])
            if c == '1':
                encoding.append(ap)
            elif c == '0':
                encoding.append(F_NOT(ap))
            else:
                raise ValueError()
        encodings.append(reduce(F_AND, encoding))

    # build "chain" between cycle steps
    cycle_constraints = []
    for idx, _ in enumerate(trace_obj.cycle_formulas):
        if idx + 1 == len(trace_obj.cycle_formulas): # last step in cycle
            next_idx = 0
        else:
            next_idx = idx + 1
        cycle_constraints.append(F_GLOBALLY(F_IMPLIES(encodings[idx], F_NEXT(F_AND(encodings[next_idx], trace_obj.cycle_formulas[next_idx])))))
    cycle_part = reduce(F_AND, cycle_constraints) # and step formulas together, add to complete formula

    # start chain
    cycle_part += F_AND(encodings[0], trace_obj.cycle_formulas[0])

    # prepend nexts to cycle
    for _ in range(len(trace_obj.prefix_formulas)):
        cycle_part = F_NEXT(cycle_part)

    # add Nexts to cycle part, add formula to check
    complete = prefix_part + cycle_part + F_NOT(formula)
    return complete


def spot_check_inclusion(formula_str, trace_str):
    formula_spot = spot.formula(formula_str)
    trace_spot = spot.parse_word(trace_str)
    formula_automaton = formula_spot.translate()
    trace_automaton = trace_spot.as_automaton()
    try:
        return spot.contains(formula_automaton, trace_automaton) # spot.contains checks whether language of its right argument is included in language of its left argument
    except RuntimeError:
        raise


def calculate_accuracy(formulas_file, traces_file, targets_file, log_file, sat_prob_file, polish, sem_desp_syn, per_size, validator, log_level, timeout, **kwargs):
    with nice_open(formulas_file, 'r') as formulas, nice_open(traces_file, 'r') as traces, nice_open(targets_file, 'r') as targets, nice_open(log_file, 'w') as log, nice_open(sat_prob_file, 'w') as sat_prob:
        spot_worker = PersistentWorker()
        line_num = 0
        tictoc = TicToc()
        if per_size:
            res = {'unsat correct' : {}, 'syntactically correct' : {}, 'only semantically correct' : {}, 'incorrect' : {}, 'invalid' : {}, 'unknown' : {}}
            def increment(key, formula_obj):
                size = formula_obj.size()
                if size in res[key]:
                    res[key][size] += 1
                else:
                    res[key][size] = 1
        else:
            res = {'unsat correct' : 0, 'syntactically correct' : 0, 'only semantically correct' : 0, 'incorrect' : 0, 'invalid' : 0, 'unknown' : 0}
            def increment(key, formula_obj):
                res[key] += 1

        for formula_str, trace_str in zip(formulas, traces):
            formula_str, trace_str = formula_str.strip(), trace_str.strip()
            line_num += 1
            target_str = next(targets).strip() if targets else None
            if target_str == '': # no trace
                target_str = None
            formula_format = 'network-' + ('polish' if polish else 'infix')
            formula_obj = ltl_parser.ltl_formula(formula_str, format=formula_format)

            if trace_str == '0': # said UNSAT
                if target_str != '':
                    if trace_str == target_str:
                        increment('unsat correct', formula_obj)
                        if log and log_level >= 4:
                            log.write("UNSAT CORRECT {:d}\ninput : {}\noutput: {}\n\n".format(line_num, formula_obj.to_str('spot'), trace_str))
                        assert not(sem_desp_syn), "Do not support for unsat"
                        continue
                    else: # incorrect
                        increment('incorrect', formula_obj)
                        target_obj = ltl_parser.ltl_trace(target_str, format=formula_format)
                        if log and log_level >= 2:
                            log.write("INCORRECT (false unsat) {:d}\ninput : {}\noutput: {}\ntarget: {}\n\n".format(line_num, formula_obj.to_str('spot'), trace_str, target_obj.to_str('spot')))
                        continue
                else: # no target available
                    raise NotImplementedError()
            else: # said SAT
                # trace valid syntactically?
                try:
                    trace_obj = ltl_parser.ltl_trace(trace_str, format=formula_format)
                except ltl_parser.ParseError as e:
                    increment('invalid', formula_obj)
                    if log and log_level >= 1:
                        log.write("INVALID {:d}\ninput  (raw): {}\noutput (raw): {}\ntarget (raw): {}\nerror: {}\n\n".format(line_num, formula_str, trace_str, target_str, e))
                    continue

                # trace equal to target (if available)?
                if target_str: # target available
                    if target_str == '0': # was unsat
                        increment('incorrect', formula_obj)
                        if log and log_level >= 2:
                            log.write("INCORRECT (false sat) {:d}\ninput : {}\noutput: {}\ntarget: {}\n\n".format(line_num, formula_obj.to_str('spot'), trace_obj.to_str('spot'), target_str))
                        continue
                    target_obj = ltl_parser.ltl_trace(target_str, format=formula_format)
                    if trace_obj.equal_to(target_obj, extended_eq=True):
                        increment('syntactically correct', formula_obj)
                        syntactically_correct = True
                        if log and log_level >= 4:
                            log.write("SYNTACTICALLY CORRECT {:d}\ninput : {}\noutput: {}\n\n".format(line_num, formula_obj.to_str('spot'), trace_obj.to_str('spot')))
                        if not sem_desp_syn:
                            continue
                    else:
                        syntactically_correct = False
                else:
                    target_obj = None
                    syntactically_correct = None

                # sat problem
                sat_obj = encode_for_satisfiability(trace_obj, formula_obj)
                sat_formula = sat_obj.to_str('spot', spacing='all ops', full_parens=True)
                if sat_prob:
                    sat_formula_conv = sat_formula.replace('1', 'True').replace('0', 'False').replace('!', '~')
                    sat_prob.write(sat_formula_conv)

                # aalta trace check
                if validator == 'aalta' or validator == 'both':
                    tictoc.tic()
                    try:
                        sat_formula_aalta = sat_obj.rewrite(Token.WEAKUNTIL).to_str('spot', spacing='all ops', full_parens=True)
                        aalta_result = aalta_sat(sat_formula_aalta, timeout=2)
                        aalta_holds = not aalta_result if aalta_result is not None else None
                    except RuntimeError as e:
                        print(str(e))
                        aalta_holds = None
                    tictoc.toc('aalta trace')
                else:
                    aalta_holds = None

                # spot trace check
                if validator == 'spot' or validator == 'both':
                    tictoc.tic()
                    finished, spot_holds = spot_worker.call(spot_check_inclusion, (formula_obj.to_str('spot', full_parens=True), trace_obj.to_str('spot')), timeout)
                    if not finished:
                        spot_holds = None
                    tictoc.toc('spot trace')
                else:
                    spot_holds = None

                # compare, evaluate trace checks
                trace_holds = aalta_holds if aalta_holds is not None else spot_holds # if both, same, else the one that is there or both None
                if validator == 'both' and aalta_holds != spot_holds:
                    print('Formula ', formula_obj.to_str('spot'))
                    print('Trace   ', trace_obj.to_str('spot'))
                    print('MISMATCH aalta: {} -- spot: {}\n'.format(aalta_holds, spot_holds))
                    log.write('MISMATCH aalta: {} -- spot: {}\n'.format(aalta_holds, spot_holds))
                    if spot_holds is not None:
                        trace_holds = spot_holds # trust spot more
                if trace_holds is None:
                    if log:
                        log.write("UNKNOWN {:d}\ninput : {}\noutput: {}\ntarget: {}\n\n".format(line_num, formula_obj.to_str('spot'), trace_obj.to_str('spot'), target_obj.to_str('spot') if target_obj else None))
                    increment('unknown', formula_obj)
                elif trace_holds:
                    if not sem_desp_syn or (not syntactically_correct):
                        if log and log_level >= 3:
                            log.write("SEMANTICALLY CORRECT {:d}\ninput : {}\noutput: {}\ntarget: {}\n\n".format(line_num, formula_obj.to_str('spot'), trace_obj.to_str('spot'), target_obj.to_str('spot') if target_obj else None))
                        increment('only semantically correct', formula_obj)
                else: # dosen't hold
                    increment('incorrect', formula_obj)
                    if log and log_level >= 2:
                        log.write("INCORRECT {:d}\ninput : {}\noutput: {}\ntarget: {}\n\n".format(line_num, formula_obj.to_str('spot'), trace_obj.to_str('spot'), target_obj.to_str('spot') if target_obj else None))
                    if sem_desp_syn and syntactically_correct:
                        raise RuntimeError('Trace is said to be syntactically correct, but does not fulfill formula!')
            log.flush()
            sys.stdout.flush()

        spot_worker.terminate()
        # tictoc.histogram(show=False) # todo: why fail?
        # evaluation
        if per_size:
            res = per_size_analysis(res, **kwargs)
        res['total'] = line_num
        res['correct'] = res['syntactically correct'] + res['only semantically correct']
        assert res['total'] == res['unsat correct'] + res['correct'] + res['incorrect'] + res['invalid'] + res['unknown']
        res_str = "Correct unsat: {:f}%, {unsat correct:d} out of {total:d}\n"\
            "Correct sat: {:f}%, {correct:d} out of {total:d}\n  Syntactically correct: {:f}%, {syntactically correct:d} out of {total:d}\n"\
            "  Semantically correct, but not syntactically: {:f}%, {only semantically correct:d} out of {total:d}\n"\
            "Incorrect: {:f}%, {incorrect:d} out of {total:d}\nInvalid: {:f}%, {invalid:d} out of {total:d}\n"\
            "Unknown: {unknown:d} out of {total:d}\n"\
            "".format(res['unsat correct'] / res['total'] * 100, res['correct'] / res['total'] * 100, res['syntactically correct'] / res['total'] * 100,
                res['only semantically correct'] / res['total'] * 100, res['incorrect'] / res['total'] * 100, res['invalid'] / res['total'] * 100, **res)
        if log and not (log is sys.stdout):
            log.write(res_str)
    return res, res_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the syntactix and semantic accuracy')
    parser.add_argument('-f', '--formulas-file', required=True)
    parser.add_argument('-t', '--traces-file', required=True)
    parser.add_argument('-r', '--targets-file', help='optional, will use trace check if not given')
    parser.add_argument('-l', '--log-file', help='file to write log about incorrect or invalid formulas to')
    parser.add_argument('-s', '--sat-prob-file', help='file to write an equivalent sat problem to')
    infix_or_polish = parser.add_mutually_exclusive_group()
    infix_or_polish.add_argument('--polish', dest='polish', action='store_true', default=True, help='Expect formulas and traces in polish notation; default True')
    infix_or_polish.add_argument('--infix', dest='polish', action='store_false', default=True, help='Expect formulas and traces in infix notation; default False')
    parser.add_argument('--sem-desp-syn', action='store_true', help='Perform semantic check even though traces matched syntactically; default False')
    parser.add_argument('--per-size', action='store_true', help='Analyze results per input formula size, otherwise do not distinguish between sizes; default False')
    parser.add_argument('--save-analysis', default=None, help='Save the plot from --per-size')
    parser.add_argument('--validator', default='both', choices=['aalta', 'spot', 'both'], help='which tool to use for semantic check; default both')
    parser.add_argument('--log-level', type=int, default='2', help='which results to log: 0=none, 1=invalid, 2=+incorrect, 3=+sem.correct, 4=+syn.correct')
    parser.add_argument('--timeout', type=int, default=10, help='timeout for semantic spot check')
    args = parser.parse_args()

    res, res_str = calculate_accuracy(**vars(args))
    print(res_str, end='')
