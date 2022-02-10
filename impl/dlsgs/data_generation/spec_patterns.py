import random
from functools import partial, reduce
import math

import numpy as np

import dlsgs.utils.ltl_parser as lp
from dlsgs.utils.ltl_parser import LiteralSet


def arity(token):
    for n, t in lp.token_dict_spot.items():
        if t == token:
            return n

class SpecPatternGen:
    def __init__(self, aps, ts):
        self.aps = aps
        self.target_length = int(random.random() * (ts[1] - ts[0]) + ts[0])
        self.params = {}
        self.params['literal_dist'] = 'normal'
        self.params['literal_dp'] = (2, 1)
        self.params['literal_interval'] = (1, min(len(aps), 10))
        self.params['var_bind_base'] = np.interp(self.target_length, [1, 10, 100], [0.8, 0.6, 0.2])
        self.params['literal_default_bind'] = 0.5
        self.params['literal_negation_prob'] = 0.5
        self.params['distort_op'] = 0.01
        self.params['state_scope_lit_cross'] = 0.2
        self.params['state_new_lit_base'] = 0.5
        self.params['prop_dp'] = (2, 3)
        self.literal_sets = set()
        self.var_collection = set()
        self._var_collection = set()
        self.state_things = []
        self.scope_things = []
    
    def update_vars(self):
        self.var_collection = self._var_collection.copy()

    def literal_set(self, bind=None, bind_to=None, negation_prob=None):
        if bind is None:
            bind=self.params['literal_default_bind']
        if bind_to is None:
            bind_to=self.var_collection
        if negation_prob is None:
            negation_prob = self.params['literal_negation_prob']
        if self.params['literal_dist'] == 'normal':
            num = np.random.normal(*self.params['literal_dp'])
            num = np.clip(num, *self.params['literal_interval'])
            num = int(num)
        else:
            raise ValueError()
        
        res = []
        available_vars_bind = list(bind_to)
        available_vars_free = set(self.aps)
        for i in range(num):
            if len(available_vars_bind) == 0:
                bind = 0
            if len(available_vars_free) == 0:
                print('(!) Bad: no more free variables for literal set')
                break
            if random.random() < bind:
                var = random.sample(available_vars_bind, 1)[0]
                available_vars_bind[:] = (q for q in available_vars_bind if q != var) # https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
            else:
                var = random.sample(available_vars_free, 1)[0]
                available_vars_free.remove(var)
            neg = random.random() < negation_prob
            res.append(lp.Literal(var, neg))
            self.var_collection.add(var)
        res = LiteralSet(res)
        self.literal_sets.add(res)
        return res

    def op(self, ch):
        n, token = lp.token_dict_spot[ch]
        # todo: randomly select other token?
        if n == 2:
            return partial(lp.LTLFormulaBinaryOp, token)
        elif n == 1:
            return partial(lp.LTLFormulaUnaryOp, token)
        else:
            raise ValueError()

    def var(self):
        bind_prop = self.params['var_bind_base'] * math.sqrt(len(self.var_collection)) # todo: more elaborate?

        if random.random() < bind_prop:
            var = random.sample(self.var_collection, 1)[0]
        else:
            var = random.sample(self.aps, 1)[0]
            self._var_collection.add(var)
        return var

    def prop(self, what='fancy', tlen=None):
        res = '(' + self._prop(tlen=tlen, what=what)[0] + ')'
        self.update_vars()
        return res

    def _prop(self, what='fancy', depth=0, tlen=None):
        if tlen is None:
            tlen = 0
            # Todo: random distortions
            if what == 'fancy':
                dp = (2, 3)
            elif what == 'broad':
                dp = (2, 2)
            elif what == 'narrow':
                dp = (2, 2)
            elif what == 'spicy':
                dp = (2, 3)
            while tlen <= 0:
                tlen = int(np.random.normal(*dp))
        assert tlen > 0
        if tlen == 1:
            return self.var(), 1

        ops = ['&', '|', '->', '<->', '!', 'X', 'G', 'F', 'U', 'W']
        if what == 'spicy':
            ps = [2, 1, 1, 1, 1, 2, 2, 2, 3, 3]
        if what == 'fancy':
            ps  = [  6,   2,    2,     2,   5,   4, 0.5, 0.5, 0.5, 0.5]
            if depth == 0:
                ps = [q * 3 if i in [2, 3] else q for (i, q) in enumerate(ps)]
        elif what == 'broad':
            ps  = [  2,   6,    4,     2,   3,   3, 0.5, 0.5, 0.5, 0.5]
        elif what == 'narrow':
            ps = [  6,   1,    1,    1,   5,   3, 0.5, 0.5, 0.5, 0.5]

        if tlen == 2:
            ps = [q if i in [4, 5, 6, 7] else 0 for (i, q) in enumerate(ps)] # eliminate binary ops
    
        op = np.random.choice(ops, p=np.array(ps)/sum(ps))
        n, _ = lp.token_dict_spot[op]

        if n == 2:
            lchild, llen = self._prop(depth=depth+1, tlen=(tlen-1)//2, what=what)
            rchild, rlen = self._prop(depth=depth+1, tlen=(tlen-1-llen), what=what)
            return '(' + lchild + ') ' + op + ' (' + rchild + ')', llen + rlen + 1
        elif n == 1:
            child, clen = self._prop(depth=depth+1, tlen=(tlen-1), what=what)
            return op + '(' + child + ')', clen + 1
        else:
            raise ValueError


    def choose_litset(self, what):
        if what == 'state':
            that, other = self.state_things, self.scope_things
        elif what == 'scope':
            that, other = self.scope_things, self.state_things
        else:
            raise ValueError
        if random.random() < self.params['state_scope_lit_cross'] and len(other) > 0:
            yes = other
        else:
            yes = that
        if len(yes) == 0:
            prob_new_lit = 1
        else:
            prob_new_lit = self.params['state_new_lit_base'] * 1/len(yes)
        if random.random() < prob_new_lit:
            lit = self.literal_set()
        else:
            lit = random.sample(yes, 1)[0]
        that.append(lit)
        lit = lit.to_formula()
        return lit



    def pattern(self, children='prop'):
        pattern_kind = np.random.choice(['absence', 'universality', 'existence', 'precedence', 'response'])
        scope_kind = np.random.choice(['globally', 'before', 'after', 'between', 'afteruntil'], p=[1/2, 1/8, 1/8, 1/8, 1/8])

        if pattern_kind == 'absence':
            if scope_kind == 'globally':
                that = 'G (! {P})'
            elif scope_kind == 'before':
                that = 'F {R} -> (!{P} U {R})'
            elif scope_kind == 'after':
                that = 'G ({Q} -> G (!{P}))'
            elif scope_kind == 'between':
                that = 'G (({Q} & !{R} & F{R}) -> (!{P} U {R}))'
            elif scope_kind == 'afteruntil':
                that = 'G ({Q} & !{R} -> (!{P} W {R}))'
        
        elif pattern_kind == 'existence':
            if scope_kind == 'globally':
                that = 'F ({P})'
            elif scope_kind == 'before':
                that = '!{R} W ({P} & ! {R})'
            elif scope_kind == 'after':
                that = 'G (!{Q}) | F ({Q} & F {P})'
            elif scope_kind == 'between':
                that = 'G ({Q} & ! {R} -> (!{R} W ({P} & !{R})))'
            elif scope_kind == 'afteruntil':
                that = 'G ({Q} & !{R} -> (!{R} U ({P} & !{R})))'
        
        elif pattern_kind == 'universality':
            if scope_kind == 'globally':
                that = 'G ({P})'
            elif scope_kind == 'before':
                that = 'F {R} -> ({P} U {R})'
            elif scope_kind == 'after':
                that = 'G ({Q} -> G {P})'
            elif scope_kind == 'between':
                that = 'G (({Q} & !{R} & F {R}) -> ({P} U {R}))'
            elif scope_kind == 'afteruntil':
                that = 'G ({Q} & !{R} -> ({P} W {R}))'
        
        elif pattern_kind == 'precedence':
            if scope_kind == 'globally':
                that = '!{P} W {S}'
            elif scope_kind == 'before':
                that = 'F {R} -> (!{P} U ({S} | {R}))'
            elif scope_kind == 'after':
                that = 'G ! {Q} | F ({Q} & (!{P} W {S}))'
            elif scope_kind == 'between':
                that = 'G (({Q} & !{R} & F {R}) -> (!{P} U ({S} | {R})))'
            elif scope_kind == 'afteruntil':
                that = 'G ({Q} & !{R} -> (!{P} W ({S} | {R})))'
        
        elif pattern_kind == 'response':
            if scope_kind == 'globally':
                that = 'G ({P} -> F {S})'
            elif scope_kind == 'before':
                that = 'F {R} -> ({P} -> (!{R} U ({S} & !{R}))) U {R}'
            elif scope_kind == 'after':
                that = 'G ({Q} -> G ({P} -> F {S}))'
            elif scope_kind == 'between':
                that = 'G (({Q} & !{R} & F {R}) -> ({P} -> (!{R} U ({S} & !{R}))) U {R})'
            elif scope_kind == 'afteruntil':
                that = 'G ({Q} & !{R} -> (({P} -> (!{R} U ({S} & !{R}))) W {R}))'

        fd = {}
        if '{P}' in that:
            fd['P'] = FL(self.choose_litset('state')) if children=='litset' else self.prop(what='fancy')
        if '{Q}' in that:
            fd['Q'] = FL(self.choose_litset('scope')) if children=='litset' else self.prop(what='broad')
        if '{R}' in that:
            fd['R'] = FL(self.choose_litset('scope')) if children=='litset' else self.prop(what='broad')
        if '{S}' in that:
            fd['S'] = FL(self.choose_litset('state')) if children=='litset' else self.prop(what='fancy')
        that = that.format(**fd)
        return that

    def ground(self, what='prop'):
        if what=='litset':
            thing = FL(self.choose_litset(random.choice(['state', 'scope'])))
        else:
            thing = self.prop('narrow')
        coin_xd = random.random()
        if coin_xd < 0.1:
            return 'G F ' + thing
        elif coin_xd < 0.2:
            return 'G ' + thing
        elif coin_xd < 0.3:
            return 'F G' + thing
        elif coin_xd < 0.4:
            return 'F ' + thing
        else:
            num_nexts = int(abs(np.random.normal(0, 3)))
            return 'X'*num_nexts + thing

    def run(self):
        current_length = 0
        self.state_things = []
        self.scope_things = []
        current_clauses = []
        while (current_length - self.target_length) < 0:
            coin_xd = random.random()
            if coin_xd < 0.4:
                pattern = self.ground()
            else:
                pattern = self.pattern()
            pattern = lp.ltl_formula(pattern)
            self.distort_ops(pattern)
            current_clauses.append(pattern)
            current_length += pattern.size()
        return reduce(lp.F_AND, current_clauses)


    def distort_ops(self, formula):
        for q in formula:
            if random.random() < self.params['distort_op']:
                if isinstance(q, lp.LTLFormulaBinaryOp):
                    q.type_ = random.sample(lp.binary_ops, 1)[0] # TODO: clean handling of realese & Co.
                elif isinstance(q, lp.LTLFormulaUnaryOp):
                    q.type_ = random.sample(lp.unary_ops, 1)[0]


def FL(f):
    return '(' + f.to_str('spot', full_parens=True) + ')'