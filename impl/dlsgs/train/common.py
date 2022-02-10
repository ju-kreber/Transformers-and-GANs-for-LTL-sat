# implementation based on DeepLTL https://github.com/reactive-systems/deepltl

from argparse import ArgumentParser
import subprocess
import os.path as path
import sys
import json
import random

import tensorflow as tf
import numpy as np

from dlsgs.utils import ltl_parser



def argparser():
    parser = ArgumentParser()
    # Meta
    parser.add_argument('--run-name', default='default', help='name of this run, to better find produced data later')
    parser.add_argument('--job-dir', default='runs', help='general job directory to save produced data into')
    parser.add_argument('--data-dir', default='datasets', help='directory of datasets')
    parser.add_argument('--ds-name', default=None, help='Name of the dataset to use')
    do_test = parser.add_mutually_exclusive_group()
    do_test.add_argument('--train', dest='test', action='store_false', default=False, help='Run in training mode, do not perform testing; default')
    do_test.add_argument('--test', dest='test', action='store_true', default=False, help='Run in testing mode, do not train')
    parser.add_argument('--binary-path', default=None, help='Path to binaries, current: aalta')
    parser.add_argument('--no-auto', action='store_true', help="Do not get parameters from params.txt when testing")
    parser.add_argument('--eval-name', default='test', help="Name of log and test files")
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--save-only', type=str, default='last', help='save which checkpoints: all, best, last')
    parser.add_argument('--params-file', type=str, help='load parameters from specified file')
    parser.add_argument('--seed', type=int, help='Global seed for python, numpy, tensorflow. If not specified, generate new one')

    # Typical Hyperparameters
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--initial-epoch', type=int, default=0, help='used to track the epoch number correctly when resuming training')
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beam-size', type=int, default=2)

    return parser


EXCLUDE_AUTO_ARGS = ['job_dir', 'run_name', 'data_dir', 'binary_path', 'test', 'force_load', 'eval_name', 'load_from', 'load_parts']


def load_params(params_dict, path, exclude_auto=True):
    with tf.io.gfile.GFile(path, 'r') as f:
        d = json.loads(f.read())
    if exclude_auto:
        for exclude in EXCLUDE_AUTO_ARGS:
            if exclude in d:
                d.pop(exclude)
    dropped = []
    for q, qm in map(lambda x: (x, '--' + x.replace('_', '-')),  list(d)):
        if any(arg.startswith(qm) for arg in sys.argv[1:]): # drop if specified on command line
            d.pop(q)
            dropped.append(qm)
    print('Loaded parameters from', path, (', dropped ' + str(dropped) + ' as specified on command line') if dropped else '')
    new =  params_dict.copy()
    new.update(d)
    return new


def setup(**kwargs):
    # If testing, load from params.txt
    if kwargs['test'] and not kwargs['no_auto']:
        if kwargs['params_file'] is not None:
            raise NotImplementedError()
        load_path = path.join(kwargs['job_dir'], kwargs['run_name'], 'params.json')
        kwargs = load_params(kwargs, load_path, exclude_auto=True)
    elif kwargs['params_file'] is not None:
        kwargs = load_params(kwargs, kwargs['params_file'], exclude_auto=False)
    binary_path = kwargs['binary_path']

    # GPU stuff
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('GPUs', gpus)
    if len(gpus) > 1:
        print("More than one GPU specified, I'm scared!")
        sys.exit(1)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Get binaries
    filenames = [] #['aalta']
    if binary_path is not None:
        for filename in filenames:
            try:
                tf.io.gfile.makedirs('bin')
                tf.io.gfile.copy(path.join(binary_path, filename), path.join('bin', filename))
            except tf.errors.AlreadyExistsError:
                pass
    
    # Random stuff
    if kwargs['seed'] is None:
        random.seed()
        kwargs['seed'] = random.randint(0, 2**32 - 1)
        print('Seed not provided, generated new one:', kwargs['seed'])
    random.seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
    tf.random.set_seed(kwargs['seed'])
    return kwargs


def log_params(job_dir, run_name, _skip=None, **kwargs):
    if _skip is None:
        _skip = []
    logdir = path.join(job_dir, run_name)
    tf.io.gfile.makedirs(logdir)
    d = kwargs.copy()
    d.update({'job_dir' : job_dir, 'run_name' : run_name})
    for _s in _skip:
        if _s in d:
            d.pop(_s)
    with tf.io.gfile.GFile(path.join(logdir, 'params.json'), 'w') as f:
        f.write(json.dumps(d, indent=4) + '\n')


def checkpoint_path(job_dir, run_name, **kwargs):
    return path.join(job_dir, run_name, 'checkpoints')


def checkpoint_callback(job_dir, run_name, save_weights_only=True, save_only='all', **kwargs):
    if save_only == 'all':
        filepath = str(path.join(checkpoint_path(job_dir, run_name), 'cp_')) + 'ep{epoch:02d}_vl{val_loss:.3f}'  # save per epoch
    elif save_only == 'best':
        filepath = str(path.join(checkpoint_path(job_dir, run_name), 'best'))  # save best only
    elif save_only == 'last':
        filepath = str(path.join(checkpoint_path(job_dir, run_name), 'last'))  # save best only
    return tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=save_weights_only, save_best_only=save_only=='best')


def get_log_dir(job_dir, run_name, **kwargs):
    return str(path.join(job_dir, run_name))

def tensorboard_callback(job_dir, run_name, **kwargs):
    log_dir = str(path.join(job_dir, run_name))
    return tf.keras.callbacks.TensorBoard(log_dir)


def last_checkpoint(job_dir, run_name, load_from=None, **kwargs):
    if load_from is not None:
        run_name = load_from
    return tf.train.latest_checkpoint(checkpoint_path(job_dir, run_name))


EVAL_PARAMS = ['ds_name', 'batch_size', 'beam_size', 'alpha', 'samples']

def test_and_analyze_ltl(pred_fn, dataset, in_vocab=None, out_vocab=None, plot_name='test_results', log_name=None, **kwargs):
    plotdir = path.join(kwargs['job_dir'], kwargs['run_name'])
    tf.io.gfile.makedirs(plotdir)
    proc_args = ['-f', '-', '-t', '-', '-r', '-', '--per-size', '--save-analysis', 'tmp_test_results', '--validator', 'aalta', '--timeout', '60', '--log-level', '4']
    if log_name is not None:
        proc_args.extend(['-l', path.join(plotdir, log_name + '.log')])
    proc = subprocess.Popen(['python3', '-m', 'dlsgs.utils.trace_check'] + proc_args,
                            stdin=subprocess.PIPE, stdout=None, stderr=None, universal_newlines=True, bufsize=10000000)
    try:
        for x in dataset:
            if kwargs['tree_pe']:
                data, pe, label = x
                pred = pred_fn([data, pe])
            else:
                data, label = x
                pred = pred_fn(data)
            if len(pred.shape) == 1:
                pred = np.expand_dims(pred, axis=0)
                data = tf.expand_dims(data, axis=0)
                label = tf.expand_dims(label, axis=0)
            for i in range(pred.shape[0]):
                label_decoded = out_vocab.decode(list(label[i, :]))
                if not label_decoded:
                    label_decoded = ''
                formula_decoded = in_vocab.decode(list(data[i, :]))
                formula_decoded = formula_decoded.replace('%', '')
                step_in = formula_decoded + '\n' + out_vocab.decode(list(pred[i, :])) + '\n' + label_decoded + '\n'
                sys.stdout.flush()
                proc.stdin.write(step_in)
                proc.stdin.flush()
    except BrokenPipeError:
        sys.exit('Pipe to trace checker broke. output:' + proc.communicate()[0])
    sys.stdout.flush()
    proc.communicate()
    tf.io.gfile.copy('tmp_test_results.png', path.join(plotdir, plot_name + '.png'), overwrite=True)
    tf.io.gfile.remove('tmp_test_results.png')
    tf.io.gfile.copy('tmp_test_results.svg', path.join(plotdir, plot_name + '.svg'), overwrite=True)
    tf.io.gfile.remove('tmp_test_results.svg')


def get_ass(lst):
    if len(lst) == 1 and lst[0] == '1':
        return {True : True}, 'True'
    if len(lst) % 2 != 0:
        raise ValueError('length of assignments not even')
    ass_it = iter(lst)
    ass_dict = {}
    for var in ass_it:
        if var in ass_dict:
            raise ValueError('Double assignment of same variable')
        val = next(ass_it)
        if val == 'True' or val == '1':
            ass_dict[var] = True
        elif val == 'False' or val == '0':
            ass_dict[var] = False
        else:
            raise ValueError('assignment var not True or False')
    s = [f'{var}={val}' for (var, val) in ass_dict.items()]
    return ass_dict, ' '.join(s)


def test_and_analyze_sat(pred_model, dataset, in_vocab, out_vocab, log_name, **kwargs):
    #from jma.data.sat_generator import spot_to_pyaiger, is_model
    import sympy.logic as syl

    logdir = path.join(kwargs['job_dir'], kwargs['run_name'])
    tf.io.gfile.makedirs(logdir)
    with open(path.join(logdir, log_name + '.log'), 'w') as log_file:
        res = {'invalid': 0, 'incorrect': 0, 'syn_correct': 0, 'sem_correct': 0}
        for x in dataset:
            if kwargs['pos_enc'] is None:
                data, label_ = x
                decodings = pred_model(data, training=False)
            else:
                data, pe, label_ = x
                decodings = pred_model([data, pe], training=False)
            for i in range(decodings.shape[0]):
                formula = in_vocab.decode(list(data[i, :]), as_list=True)
                pred = out_vocab.decode(list(decodings[i, :]), as_list=True)
                label = out_vocab.decode(list(label_[i, :]), as_list=True)
                formula_obj = ltl_parser.ltl_formula(''.join(formula), 'network-polish')
                formula_str = formula_obj.to_str('spot')
                _, pretty_label_ass = get_ass(label)
                try:
                    ass, pretty_ass = get_ass(pred)
                except ValueError as e:
                    res['invalid'] += 1
                    msg = f"INVALID ({str(e)})\nFormula: {formula_str}\nPred:     {' '.join(pred)}\nLabel:    {pretty_label_ass}\n"
                    log_file.write(msg)
                    continue
                if pred == label:
                    res['syn_correct'] += 1
                    msg = f"SYNTACTICALLY CORRECT\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:    {pretty_label_ass}\n"
                    # log_file.write(msg)
                    continue

                # semantic checking
                formula_sympy = formula_obj.to_sympy()
                try:
                    substituted = syl.simplify_logic(formula_sympy.subs(ass))
                    holds = substituted == syl.true
                except KeyError as e:
                    res['incorrect'] += 1
                    msg = f"INCORRECT (var {str(e)} not in formula)\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:  {pretty_label_ass}\n"
                    log_file.write(msg)
                    continue
                if holds:
                    res['sem_correct'] += 1
                    msg = f"SEMANTICALLY CORRECT\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:  {pretty_label_ass}\n"
                    log_file.write(msg)
                else:
                    res['incorrect'] += 1
                    msg = f"INCORRECT\nFormula: {formula_str}\nPred:    {pretty_ass}\nLabel:   {pretty_label_ass}\nRemaining formula: {substituted}\n"
                    log_file.write(msg)

        total = sum(res.values())
        correct = res['syn_correct'] + res['sem_correct']
        msg = (f"Correct: {correct/total*100:.1f}%, {correct} out of {total}\nSyntactically correct: {res['syn_correct']/total*100:.1f}%\nSemantically correct: {res['sem_correct']/total*100:.1f}%\n"
               f"Incorrect: {res['incorrect']/total*100:.1f}%\nInvalid: {res['invalid']/total*100:.1f}%\n")
        log_file.write(msg)
        print(msg, end='')
    with tf.io.gfile.GFile(path.join(logdir, log_name + '_params.json'), 'w') as f:
        d = { k : v for k, v in kwargs.items() if k in EVAL_PARAMS}
        f.write(json.dumps(d, indent=4) + '\n')
