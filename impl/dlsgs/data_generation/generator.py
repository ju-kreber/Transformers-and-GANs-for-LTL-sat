#!python3.7

# implementation based on DeepLTL https://github.com/reactive-systems/deepltl

# pylint: disable=line-too-long

from __future__ import generator_stop  # just to be safe with python 3.7
import sys, os, re
import signal
import datetime
import argparse
import random
import json

from tensorflow.io import gfile # pylint: disable=import-error

from dlsgs.utils import ltl_parser, utils
from dlsgs.data_generation.ltl import solve_ltl
from dlsgs.data_generation.prop import solve_prop


class DistributionGate():
    # interval: [a, b]
    def __init__(self, key, distribution, interval, total_num, **kwargs):
        # optional: start_calc_at together with alpha
        self.dist = {}
        self.targets = {}
        self.fulls = {}
        self.key = key
        self.interval = interval
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.0
        self.distribution = distribution
        bleft, bright = interval
        if key == 'formula size':
            self.bins = list(range(bleft, bright + 1))
            self.get_val = lambda x: x.size()
        else:
            raise ValueError()
        for b in self.bins:
            self.dist[b] = 0
        if distribution == 'uniform':
            if 'start_calc_from' in kwargs:
                start = kwargs['start_calc_from']
                self.enforced_bins = list(
                    filter(lambda x: x >= start, self.bins))
            else:
                self.enforced_bins = self.bins
            num_actual_bins = len(self.enforced_bins)
            for b in self.bins:
                self.targets[b] = total_num * \
                    (1 - self.alpha) / num_actual_bins
                self.fulls[b] = self.dist[b] >= self.targets[b]
        elif distribution == 'arbitrary':
            pass
        else:
            raise ValueError()

    def gate(self, formula_obj: ltl_parser.LTLFormula) -> bool:
        val = self.get_val(formula_obj)
        if val < self.interval[0] or val > self.interval[1]:  # not in range
            return False
        if self.distribution == 'arbitrary':
            return True
        else:
            return not self.fulls[val]

    def update(self, formula_obj: ltl_parser.LTLFormula):
        val = self.get_val(formula_obj)
        if val >= self.interval[0] and val <= self.interval[1]:
            self.dist[val] += 1
            if self.distribution != 'arbitrary' and self.dist[val] >= self.targets[val]:
                self.fulls[val] = True

    def histogram(self, show=True, save_to=None):
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(1)
        counts = [val for key, val in sorted(self.dist.items())]
        axis.bar(self.bins, counts, width=1,
                 color='#3071ff', edgecolor='white')
        axis.set_ylabel('number of items')
        axis.set_xlabel(self.key)
        axis.title.set_text('alpha = ' + str(self.alpha))
        if save_to is not None:
            figure.savefig(save_to)
        if show:
            plt.show()
        else:
            plt.close(figure)

    def full(self) -> bool:
        if self.distribution == 'arbitrary':
            return False
        else:
            return all([self.fulls[eb] for eb in self.enforced_bins])


def generate_examples(params, pers_worker):
    interrupted = False
    def signal_handler(signal, frame):
        nonlocal interrupted
        print(f"Received signal {signal:d}, interrupting generation")
        interrupted = True
    signal.signal(signal.SIGINT, signal_handler)

    if params.token_format['variables'] == 'alphabetical':
        if params.num_variables > 26:
            raise ValueError("Cannot generate more than 26 APs")
        variables = list(map(chr, range(97, 97 + params.num_variables)))
    elif params.token_format['variables'] == 'p_numeric':
        variables = list(f'p{q:d}' for q in range(params.num_variables))
    if not isinstance(params.tree_size, tuple):
        params.tree_size = (1, params.tree_size)

    if params.formula_generator == 'spot':
        import spot
        token_dist = re.sub(r'ap=[!]', 'ap={}'.format(len(variables)), params.token_dist)
        if params.problem == 'ltl':
            formula_generator = spot.randltl(variables, seed=params.seed, tree_size=params.tree_size, output='ltl', ltl_priorities=token_dist, simplify=0)
        elif params.problem == 'prop':
            formula_generator = spot.randltl(variables, seed=params.seed, tree_size=params.tree_size, output='bool', boolean_priorities=token_dist, simplify=0)
        else:
            raise ValueError()
    elif params.formula_generator == 'spec_patterns':
        assert params.problem == 'ltl'
        from dlsgs.data_generation.spec_patterns import SpecPatternGen
        class SpecPatternGenWrapper:
            def __next__(self):
                d = SpecPatternGen(variables, params.tree_size)
                return d.run()
        formula_generator = SpecPatternGenWrapper()
    elif params.formula_generator == 'dac_patterns':
        assert params.problem == 'ltl'
        assert params.token_format['variables'] == 'alphabetical'
        from dlsgs.data_generation.general_patterns import GeneralPatternGenerator
        formula_generator = GeneralPatternGenerator('dac', variables, params.tree_size)
    else:
        raise ValueError()

    tictoc = utils.TicToc()
    dist_gate = DistributionGate('formula size', params.formula_dist, params.tree_size, params.num_examples, start_calc_from=10, alpha=params.alpha)

    # generate samples
    print('Generating examples...')
    examples = []
    timeout_formulas = []
    sat_examples = 0
    unsat_examples = 0
    total_examples = 0
    dropped_sat_examples = 0
    dropped_unsat_examples = 0
    dropped_dist_examples = 0
    dropped_timeout_examples = 0
    time_started = datetime.datetime.now()
    last_msg_time = time_started
    last_msg_percent = 0
    accu = { k : 0 for k in {'model_count', 'model_poss', 'model_frac', 'log_model_count', 'log_model_poss', 'log_model_frac', 'in_length', 'out_length', 'solve_time'}}
    if params.include_solver_statistics and params.problem == 'ltl' and params.ltl_solver == 'leviathan':
        accu.update({k : 0 for k in ['lev_evtls', 'lev_frames', 'lev_steps', 'lev_max_model_size', 'lev_max_depth']})
    info = {'max_in_length' : 0, 'max_out_length' : 0}
    while True:
        current_percent = total_examples / params.num_examples * 100
        now = datetime.datetime.now()
        if current_percent - last_msg_percent >= params.log_each_x_percent or now - last_msg_time > datetime.timedelta(hours=1):
            last_msg_percent = current_percent
            last_msg_time = now
            print("Generated {:,d} of {:,d} examples ({:4.1f}%); dropped {:,d} sat, {:,d} unsat, {:,d} dist, {:,d} timeout; at {:s} runtime".format(total_examples, 
              params.num_examples, current_percent, dropped_sat_examples, dropped_unsat_examples, dropped_dist_examples, dropped_timeout_examples, utils.strfdelta_hms(now - time_started)))
            sys.stdout.flush()
        if total_examples >= params.num_examples:
            #print(f'Terminated: Generated specified amount of examples ({total_examples}).')
            break
        if dist_gate.full():
            #print(f'Terminated: Distribution is full.')
            break
        if interrupted:
            #print(f'Terminated: Interrupted.')
            break
        if params.max_runtime != 0.0 and (now - time_started).total_seconds() > params.max_runtime:
            print('Exiting due to exceeded runtime')
            break

        tictoc.tic()
        formula = next(formula_generator)
        if not isinstance(formula, ltl_parser.LTLFormula):
            if not isinstance(formula, str):
                if formula is None:
                    continue # only case: dac_patterns max_size too small
                formula = formula.to_str()
            formula_obj = ltl_parser.ltl_formula(formula, 'spot')
        else:
            formula_obj = formula
        tictoc.toc('formula generation')

        if not dist_gate.gate(formula_obj):  # formula doesn't fit distribution
            dropped_dist_examples += 1
            continue
        if formula == '1': # special case, currently not handled by all solvers (leviathan)
            continue

        tictoc.tic()
        if params.problem == 'ltl':
            is_sat, witness, d = solve_ltl(formula_obj, params.ltl_solver, pers_worker, timeout=params.timeout, simplify=False, no_disjunctions=True, witness='witness' in params.subproblem, binary_path=params.binary_dir)
        elif params.problem == 'prop':
            is_sat, witness, d = solve_prop(formula_obj, params.prop_solver, params.solution_choice, simplify=True, count_models=params.include_log_model_count, model_counting=params.model_counting, binary_path=params.binary_dir)
        tictoc.toc('solving')

        if is_sat is None:  # due to timeout
            # print('Trace generation timed out ({:d}s) for formula {}'.format(int(timeout), formula_obj.to_str('spot')))
            if params.require_solved:
                if params.save_timeouts:
                    timeout_formulas.append(formula_obj.to_str('spot', spacing='all ops', full_parens=True))
                dropped_timeout_examples += 1
                continue
            else:  # no solved required
                # TODO how to count? for now, just do not increment specifc counter (only total at end)
                witness_str = ''
                dist_gate.update(formula_obj)
        elif not is_sat and params.require_sat:
            dropped_unsat_examples += 1
            continue
        elif not is_sat and not params.require_sat: # unsat
            if (params.frac_unsat is not None) and unsat_examples >= params.frac_unsat * params.num_examples:
                dropped_unsat_examples += 1
                continue
            else:  # more unsat samples needed
                witness_str = '0'
                dist_gate.update(formula_obj)
                unsat_examples += 1
        else:  # is_sat
            if not params.require_sat and (params.frac_unsat is not None) and sat_examples >= (1 - params.frac_unsat) * params.num_examples:
                dropped_sat_examples += 1
                continue
            elif random.random() < params.drop_sat_prob:
                # don't log
                continue
            else:  # more sat samples needed
                if params.problem == 'ltl':
                    if 'witness' in params.subproblem:
                        witness_str = witness.to_str('network-' + params.operator_notation)
                    else:
                        witness_str = '1'
                    out_length = len(witness_str)
                elif params.problem == 'prop':
                    assert not params.solution_choice == 'all'
                    # todo: format in case of trivial?
                    if True in witness:
                        witness_str = '1'
                    else:
                        witness_str = ''.join([var.name + str(int(val)) for (var, val) in sorted(witness.items(), key=lambda x:x[0].name)])
                    out_length = len(witness_str)
                info['max_out_length'] = max(info['max_out_length'], out_length)
                accu['out_length'] += out_length
                dist_gate.update(formula_obj)
                sat_examples += 1

        for k, v in d.items():
            if k in accu:
                accu[k] += v
        in_length = formula_obj.size()
        info['max_in_length'] = max(info['max_in_length'], in_length)
        accu['in_length'] += in_length
        formula_str = formula_obj.to_str('network-' + params.operator_notation)
        examples.append((formula_str, witness_str, d))
        total_examples += 1

    if params.problem == 'prop' and params.include_log_model_count:
        print('sat ex', sat_examples)
        info['avg_model_count'] = accu['model_count'] / sat_examples
        info['avg_model_poss'] = accu['model_poss'] / sat_examples
        info['avg_model_percent'] = accu['model_count'] / accu['model_poss'] * 100
        info['avg_model_pre_percent'] = accu['model_frac'] / sat_examples * 100
        print('Average model count: {avg_model_count:.1f} / {avg_model_poss:.1f} (post {avg_model_percent:.1f}%, pre {avg_model_pre_percent:.1f}%)'.format(**info))
        info['avg_log_model_count'] = accu['log_model_count'] / sat_examples
        info['avg_log_model_poss'] = accu['log_model_poss'] / sat_examples
        info['avg_log_model_percent'] = accu['log_model_count'] / accu['log_model_poss'] * 100
        info['avg_log_model_pre_percent'] = accu['log_model_frac'] / sat_examples * 100
        print('Average log model count: {avg_log_model_count:.1f} / {avg_log_model_poss:.1f} (post {avg_log_model_percent:.1f}%; pre {avg_log_model_pre_percent:.1f}%)'.format(**info))
    if params.include_solve_time:
        info['avg_solve_time'] = accu['solve_time'] / total_examples
        print('Average solve time: {avg_solve_time:.2f} ms'.format(**info))
    if params.include_solver_statistics and params.problem == 'ltl' and params.ltl_solver == 'leviathan':
        for k in ['lev_evtls', 'lev_frames', 'lev_steps', 'lev_max_model_size', 'lev_max_depth']:
            info['avg_' + k] = accu[k] / total_examples
        print('Average leviathan statistics: {avg_lev_evtls:.1f} eventualities, {avg_lev_frames:.1f} frames, {avg_lev_steps:.1f} steps, {avg_lev_max_model_size:.1f} max model size, {avg_lev_max_depth:.1f} max depth'.format(**info))
    info['avg_in_length'] = accu['in_length'] / total_examples
    info['avg_out_length'] = accu['out_length'] / sat_examples
    info['runtime'] = utils.strfdelta_hms(datetime.datetime.now() - time_started)
    info['_dist_gate'] = dist_gate
    info['_tictoc'] = tictoc
    info['examples_generated'] = total_examples
    info['examples_generated_sat'] = sat_examples
    info['examples_generated_unsat'] = unsat_examples
    print('Average formula length {avg_in_length:.1f} and witness length {avg_out_length:.1f}'.format(**info))
    print('Generated {:d} examples ({:d} sat, {:d} unsat). {:d} requested.'.format(total_examples, sat_examples, unsat_examples, params.num_examples))
    return examples, timeout_formulas, info


def split_and_write(examples, timeouts, params, log_dict):
    random.Random(params.seed).shuffle(examples)
    num_examples = len(examples)
    res = {}
    total_val = sum(params.splits.values())
    current_val = 0.0
    for split, val in params.splits.items():
        res[split] = examples[int(current_val/total_val * num_examples) : int((current_val + val)/total_val * num_examples)]
        current_val += val

    assert params.file_format == 'text/2lines'
    if params.include_simplified_formula:
        log_dict['file_format'] += '+simplified_formula'
    if params.include_log_model_count:
        log_dict['file_format'] += '+log_model_count'
    try:
        if params.solution_choice == 'all':
            log_dict['file_format'] += '+all_solutions'
    except AttributeError:
        pass # so sorry
    if params.include_solve_time:
        log_dict['file_format'] += '+solve_time'
    if params.include_solver_statistics:
        log_dict['file_format'] += '+solver_statistics'

    print(f'Writing dataset of {num_examples} to {params.output_dir}...')
    gfile.makedirs(params.output_dir)
    for split, data in res.items():
        path = os.path.join(params.output_dir, split + '.txt')
        with gfile.GFile(path, 'w') as f:
            for ex_in, ex_out, d in data:
                f.write(ex_in)
                if params.include_simplified_formula:
                    f.write(' #simplified_formula: ' + d['simplified_formula'].to_str('network-'+ params.operator_notation))
                if params.include_log_model_count and ex_out != '0':
                    f.write(' #log_model_count: {log_model_count:.1f} / {log_model_poss:d}'.format(**d))
                if params.include_solve_time:
                    f.write(' #solve_time: {solve_time:.2f}'.format(**d))
                if params.include_solver_statistics:
                    for k, v in d.items():
                        if k.startswith(params.ltl_solver[:3]) or k.startswith(params.prop_solver[:3]):
                            f.write(' #{}: {}'.format(k, v))
                f.write('\n' + ex_out + '\n')
    log_dict['_dist_gate'].histogram(show=False, save_to='tmp_dist.png')      # For distribution analysis
    gfile.copy('tmp_dist.png', os.path.join(params.output_dir, 'dist.png'), overwrite=True)
    gfile.remove('tmp_dist.png')
    del log_dict['_dist_gate']
    log_dict['_tictoc'].histogram(show=False, save_to='tmp_timing.png')         # For timing analysis
    gfile.copy('tmp_timing.png', os.path.join(params.output_dir, 'timing.png'), overwrite=True)
    gfile.remove('tmp_timing.png')
    del log_dict['_tictoc']
    with gfile.GFile(os.path.join(params.output_dir, 'info.json'), 'w') as f:
        log_dict['timestamp'] = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write(json.dumps(log_dict, indent=4) + '\n')

    if params.save_timeouts:
        with gfile.GFile(os.path.join(params.output_dir, 'timeouts.txt'), 'w') as f:
            f.write('\n\n'.join(timeouts) + '\n\n')


def run():
    # Argument processing
    parser = argparse.ArgumentParser(description='Data generator. Supports multiple problems and formats.')
    parser.add_argument('--output-dir', '-od', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--splits', default='train:8,val:1,test:1')
    parser.add_argument('--problem', required=True, choices=['prop', 'ltl'])
    parser.add_argument('--subproblem', choices=['witness', 'decision+witness', 'decision'], default='decision+witness')
    parser.add_argument('--operator-notation', choices=['infix', 'polish'], default='polish')
    parser.add_argument('--token-format', default='variables:alphabetical,constants:numerical,operators:character,traces:classic')
    parser.add_argument('--ltl-solver', type=str, choices=['spot', 'aalta', 'leviathan'], default='spot', help='which tool to get a trace (or unsat) from; default spot')
    parser.add_argument('--prop-solver', type=str, choices=['sympy', 'pyaiger'], default='sympy')
    parser.add_argument('--timeout', type=float, default=10, help='time in seconds to wait for the generation of a single example')
    parser.add_argument('--file-format', default='text/2lines')

    require_solved = parser.add_mutually_exclusive_group()
    require_solved.add_argument('--require-solved', dest='require_solved', action='store_true', default=True, help='require a trace to be found for each formula (useful for training/testing set); default')
    require_solved.add_argument('--allow-unsolved', dest='require_solved', action='store_false', default=True, help='allow formulas without a corresponding trace found (useful for further evaluation)')
    require_sat = parser.add_mutually_exclusive_group()
    require_sat.add_argument('--require-sat', dest='require_sat', action='store_true', default=True)
    require_sat.add_argument('--allow-unsat', dest='require_sat', action='store_false', default=True)
    parser.add_argument('--num-variables', '-nv', type=int, default=5)
    parser.add_argument('--num-examples', '-ne', type=int, default=1000)
    parser.add_argument('--tree-size', '-ts', type=str, default='15', metavar='MAX_TREE_SIZE', help="Maximum tree size of generated formulas. Range can be specified as 'MIN-MAX'; default minimum is 1")
    parser.add_argument('--formula-dist', type=str, default='arbitrary')
    parser.add_argument('--alpha', type=float, default=0.0, help='Distribution parameter')
    parser.add_argument('--drop-sat-prob', type=float, default=0.0)
    parser.add_argument('--frac-unsat', type=str, default='0.5')
    parser.add_argument('--fsa-dist', default='arbitrary', help="the Fraction of Satisfying Assignments maybe should to be constrained")
    parser.add_argument('--token-dist', default=None)
    parser.add_argument('--formula-generator', choices=['spot', 'spec_patterns', 'dac_patterns'], default='spot')
    parser.add_argument('--generater-dist-args', default=None)
    parser.add_argument('--solution-choice', choices=['first', 'all', 'random'], default='first')
    parser.add_argument('--include-simplified-formula', action='store_true')
    parser.add_argument('--include-log-model-count', action='store_true')
    parser.add_argument('--include-solve-time', action='store_true')
    parser.add_argument('--include-solver-statistics', action='store_true')
    parser.add_argument('--model-counting', choices=['naive', 'approximate'], default='approximate')
    parser.add_argument('--binary-dir', default='./bin')
    parser.add_argument('--save-timeouts', action='store_true')
    parser.add_argument('--max-runtime', type=float, default=0.0)

    parser.add_argument('--log-each-x-percent', type=float, default=1.0)
    parser.add_argument('--comment', '-C', type=str)

    args = parser.parse_args()
    original_args = argparse.Namespace(**vars(args))

    if '-' in args.tree_size:
        args.tree_size = tuple(map(int, args.tree_size.split('-')))
    else:
        args.tree_size = int(args.tree_size)
    token_format = {'variables' : 'alphabetical', 'constants' : 'numerical', 'operators' : 'character', 'traces' : 'classic'}
    token_format.update(dict([q.strip().split(':') for q in args.token_format.split(',')]))
    args.token_format = token_format
    args.splits = { k : int(v) for k, v in [q.strip().split(':') for q in args.splits.split(',')] }
    if args.token_dist is None:
        if args.problem == 'ltl':
            args.token_dist = 'ap=!,false=1,true=1,not=1,F=1,G=1,X=1,equiv=0,implies=1,xor=0,R=0,U=1,W=1,M=0,and=1,or=1'
        elif args.problem == 'prop':
            args.token_dist = 'ap=!,false=1,true=1,not=1,equiv=0,implies=1,xor=0,and=1,or=1'
        original_args.token_dist = args.token_dist
    if args.save_timeouts and not args.require_solved:
        sys.exit('--save-timeouts doesnt make sense without --require-solved')
    if args.require_sat == ('decision' in args.subproblem):
        sys.exit('--subproblem and --allow-unsat do not match')
    if args.formula_generator != 'spot':
        del args.token_dist
    if args.problem != 'prop':
        del args.prop_solver
    if args.problem != 'ltl':
        del args.ltl_solver
    if not 'witness' in args.subproblem:
        args.solution_choice = None
    if args.frac_unsat.lower() == 'none':
        args.frac_unsat = None
    else:
        args.frac_unsat = float(args.frac_unsat)


    log_dict = vars(original_args)
    pers_worker = utils.PersistentWorker()
    # todo copy in case of gs://...
    ld_path = os.environ.get('LD_LIBRARY_PATH', None)
    os.environ['LD_LIBRARY_PATH'] = args.binary_dir + (':' + ld_path if ld_path is not None else '') # TODO only works on linux..
    try:
        examples, timeouts, stats = generate_examples(args, pers_worker)
    finally:
        pers_worker.terminate()
    log_dict.update(stats)
    split_and_write(examples, timeouts, args, log_dict)

if __name__ == '__main__':
    run()
