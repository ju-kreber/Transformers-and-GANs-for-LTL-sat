import os
import multiprocessing as mp

import tensorflow as tf
import numpy as np

from dlsgs.utils import ltl_parser, math_parser
from dlsgs.data_generation.ltl import aalta_sat
from dlsgs.utils.math_parser import parse_help

class ReplayBuffer():
    def __init__(self, params, update_method='reservoir'):
        self.buffer_size = 10000
        shape = (self.buffer_size, params['max_encode_length'], params['d_embed_enc'])
        self.buffer = tf.Variable(tf.zeros(shape, dtype=tf.float32), trainable=False)
        self.mask_buffer = tf.Variable(tf.zeros(shape[:2], dtype=tf.int32), trainable=False)
        self.buffer_items = 0
        self.total_items = 0
        self.ring_pointer = 0
        self.update_method = update_method
        print('using replay buffer of size', self.buffer_items, 'update method', self.update_method)
    
    def update(self, tensor, mask_tensor): # implements simple reservoir sampling
        num_items = tf.shape(tensor)[0]
        self.total_items += num_items
        num_to_full = self.buffer_size - self.buffer_items
        if num_to_full > 0:
            assert self.ring_pointer == self.buffer_items
            num_filling = min(num_to_full, num_items)
            filling = tensor[:num_filling]
            mask_filling = mask_tensor[:num_filling]
            to_update = self.buffer[self.buffer_items:self.buffer_items+num_filling]
            mask_to_update = self.mask_buffer[self.buffer_items:self.buffer_items+num_filling]
            to_update.assign(filling)
            mask_to_update.assign(mask_filling)
            self.buffer_items += num_filling
            self.ring_pointer += num_filling
            num_left = num_items - num_filling
        else:
            num_left = num_items
        if num_left > 0:
            tensor = tensor[:num_left] # most often a no-op
            mask_tensor = mask_tensor[:num_left]
            update_method, *update_params = self.update_method.split('$')
            if update_method == 'reservoir':
                ps = tf.random.uniform([num_left], 0, self.total_items, dtype=tf.int32) # not fully correct due to batch, but whatever :D
                do_replace = ps < self.buffer_size
                num_replace = tf.reduce_sum(tf.cast(do_replace, tf.int32))
                replace_indices = tf.boolean_mask(ps, do_replace)
            elif update_method == 'constant':
                ps = tf.random.uniform([num_left], 0, 1, dtype=tf.float32)
                do_replace = ps < float(update_params[0])
                num_replace = tf.reduce_sum(tf.cast(do_replace, tf.int32))
                replace_indices = tf.random.uniform([num_replace], 0, self.buffer_size, dtype=tf.int32)
            else:
                raise ValueError()
            if num_replace > 0:
                replace_values = tf.boolean_mask(tensor, do_replace)
                replace_mask_values = tf.boolean_mask(mask_tensor, do_replace)
                self.buffer.scatter_nd_update(replace_indices, replace_values)
                self.mask_buffer.scatter_nd_update(replace_indices, replace_mask_values)

    def get(self, how_many):
        assert how_many <= self.buffer_items
        indices = tf.random.uniform([how_many], 0, self.buffer_items, dtype=tf.int32)
        return tf.gather(self.buffer[:self.buffer_items], indices, axis=0), tf.gather(self.mask_buffer[:self.buffer_items], indices, axis=0)

    def filled(self):
        return self.buffer_items == self.buffer_size



class CreatedBuffer():
    def __init__(self, params):
        self.update_method = params['gan_created_buffer_method']
        self.buffer_size = params['gan_created_buffer_size']
        shape = (self.buffer_size, params['max_encode_length'] + 1)
        self.buffer = tf.Variable(tf.zeros(shape, dtype=tf.int32), trainable=False)
        self.buffer_items = 0
        self.total_items = 0
        self.ring_pointer = 0
        print('using creation buffer of size', self.buffer_size, 'update method', self.update_method)
    
    def update(self, tensor): # implements simple reservoir sampling
        num_items = tf.shape(tensor)[0]
        self.total_items += num_items
        num_to_full = self.buffer_size - self.buffer_items
        if num_to_full > 0:
            assert self.ring_pointer == self.buffer_items
            num_filling = min(num_to_full, num_items)
            filling = tensor[:num_filling]
            to_update = self.buffer[self.buffer_items:self.buffer_items+num_filling]
            to_update.assign(filling)
            self.buffer_items += num_filling
            self.ring_pointer += num_filling
            num_left = num_items - num_filling
        else:
            num_left = num_items
        if num_left > 0:
            tensor = tensor[:num_left] # most often a no-op
            update_method, *update_params = self.update_method.split('$')
            if update_method == 'reservoir':
                ps = tf.random.uniform([num_left], 0, self.total_items, dtype=tf.int32) # not fully correct due to batch, but whatever :D
                do_replace = ps < self.buffer_size
                num_replace = tf.reduce_sum(tf.cast(do_replace, tf.int32))
                replace_indices = tf.boolean_mask(ps, do_replace)
            elif update_method == 'constant':
                ps = tf.random.uniform([num_left], 0, 1, dtype=tf.float32)
                do_replace = ps < float(update_params[0])
                num_replace = tf.reduce_sum(tf.cast(do_replace, tf.int32))
                replace_indices = tf.random.uniform([num_replace], 0, self.buffer_size, dtype=tf.int32)
            else:
                raise ValueError()
            if num_replace > 0:
                replace_values = tf.boolean_mask(tensor, do_replace)
                self.buffer.scatter_nd_update(replace_indices, replace_values)

    def get(self, how_many):
        assert how_many <= self.buffer_items
        indices = tf.random.uniform([how_many], 0, self.buffer_items, dtype=tf.int32)
        return tf.gather(self.buffer[:self.buffer_items], indices, axis=0)

    def filled(self):
        return self.buffer_items == self.buffer_size


def parse_polish(tokens):
    if len(tokens) == 0:
        return False, 0, []
    num_children, type_, *name = tokens.pop(0)
    if num_children == 2:
        cont, ll, tokens = parse_polish(tokens)
        if cont:
            cont, lr, tokens = parse_polish(tokens)
        else:
            lr = 0
        return cont, ll+lr+1, tokens
    elif num_children == 1:
        cont, l, tokens = parse_polish(tokens)
        return cont, l+1, tokens
    elif num_children == 0:
        return True, 1, tokens
    elif num_children == -1:
        return False, 1, tokens
    else:
        raise ValueError("Illegal token '" + str(type_) + "'")


def parse_score(s):
    num_removed = s.count('%') # replace actually illegal tokens
    s = s.replace('%', '')
    try:
        tokens, stats = ltl_parser.tokenize_formula(s, 'network', return_statistics=True)
    except ltl_parser.ParseError as e:
        return None, None, None
    vals = np.array(list(stats.values()))
    rel = vals / vals.sum()
    entropy = -np.sum(rel * np.log(rel))
    original_length = len(tokens) + num_removed
    fragments = 0
    valid_coverage = 0.
    while tokens:
        fragments += 1
        ok, l, tokens = parse_polish(tokens)
        if ok:
            valid_coverage += l / original_length
            if len(tokens) > 0 and tokens[0][1] == ltl_parser.Token.EOS:
                tokens.pop(0)
                valid_coverage += 1 / original_length
    return entropy, fragments, valid_coverage


def parse_score_math(tokens):
    full_length = len(tokens)
    current_length = full_length
    coverage_num = 0
    fragments = 0
    while current_length > 0:
        fragments += 1
        try:
            f, tokens = parse_help(tokens)
            last_length = current_length
            current_length = len(tokens)
            coverage_num += last_length - current_length
            if current_length > 0 and tokens[0] == '<eos>':
                tokens.pop(0)
                current_length -= 1
                coverage_num += 1
        except ltl_parser.ParseError:
            break
    return fragments, coverage_num / full_length



def get_corrects(candidates, params, step=None, entropies=None):
    if not params['gan_generate_confusion']:
        entropies = None # no point in printing them
    
    if params['problem'] == 'ltl':
        assert params['gan_solve_tool'] == 'aalta'
        assert params['gan_solve_valid_samples']
        candidates = [''.join(q) for q in candidates]

        with mp.Pool(64) as p:
            results = p.map(_parse_and_solve_ltl_aalta_fun, enumerate(candidates))

        solved_indices_sat, solved_formulas_sat, solved_indices_unsat, solved_formulas_unsat = [], [], [], []
        # TODO: also keep undecided ones? timeout, error?
        for index, formula_str, sat in results:
            if sat is True:
                solved_indices_sat.append(index)
                solved_formulas_sat.append(formula_str)
            elif sat is False:
                solved_indices_unsat.append(index)
                solved_formulas_unsat.append(formula_str)
        min_len = min(len(solved_indices_sat), len(solved_indices_unsat))
        if params['gan_balance_valid_samples']:
            solved_indices_sat, solved_formulas_sat = solved_indices_sat[:min_len], solved_formulas_sat[:min_len]
            solved_indices_unsat, solved_formulas_unsat = solved_indices_unsat[:min_len], solved_formulas_unsat[:min_len]

        if params['gan_save_valid_samples']:
            with open(os.path.join(params['job_dir'], params['run_name'], 'valid_samples.txt'), 'a') as save_file:
                if step is not None:
                    save_file.write('#step ' + str(step) + '\n')
                for index, formula_str in zip(solved_indices_sat, solved_formulas_sat):
                    save_file.write(formula_str + (' #classifier_entropy:'+str(entropies[index]) if entropies is not None else '') + '\n' + '1\n')
                for index, formula_str in zip(solved_indices_unsat, solved_formulas_unsat):
                    save_file.write(formula_str + (' #classifier_entropy:'+str(entropies[index]) if entropies is not None else '') + '\n' + '0\n')

        return solved_indices_sat, solved_formulas_sat, solved_indices_unsat, solved_formulas_unsat
    
    elif params['problem'] == 'math':
        with mp.Pool(64) as p:
            results = p.map(_parse_math_fun, enumerate(candidates))
        if params['gan_save_valid_samples']:
            with open(os.path.join(params['job_dir'], params['run_name'], 'valid_samples.txt'), 'a') as save_file:
                if step is not None:
                    save_file.write('#step ' + str(step) + '\n')
                for index, s in results:
                    if s is not None:
                        save_file.write(s + '\n')
        return None, None, None, None
        

def _parse_and_solve_ltl_aalta_fun(index_candidate):
    index, candidate = index_candidate
    candidate = candidate.replace('%', '') # replace actually illegal tokens
    try:
        tokens = ltl_parser.tokenize_formula(candidate, 'network')
    except ltl_parser.ParseError:
        return index, None, None
    try:
        ltl_formula, remainder = ltl_parser.parse_polish_formula(tokens)
    except ltl_parser.ParseError:
        return index, None, None
    if not all([q[0] == -1 for q in remainder]):
        return index, None, None
    
    formula_str = ltl_formula.rewrite(ltl_parser.Token.WEAKUNTIL).to_str('spot', spacing='all ops', full_parens=True) # unambiguous form
    try:
        sat = aalta_sat(formula_str, timeout=5, mute=True)
    except RuntimeError:
        sat = None
    except UnicodeDecodeError:
        sat = None
    return index, ltl_formula.to_str('network-polish'), sat


def _parse_math_fun(index_candidate):
    index, candidate = index_candidate
    try:
        node, remainder = math_parser.parse_help(candidate)
    except ltl_parser.ParseError:
        return index, None
    if len(remainder) > 0:
        return index, None
    return index, node.to_str('network-polish')
