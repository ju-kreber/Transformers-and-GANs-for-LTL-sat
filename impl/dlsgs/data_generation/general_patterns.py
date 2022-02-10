import random
import re

import spot
import spot.gen as sg

from dlsgs.utils import ltl_parser

class GeneralPatternGenerator():
    def __init__(self, pattern_type, variables, tree_size):
        self.variables = variables
        self.tree_size = tree_size
        if pattern_type == 'dac':
            gen = sg.ltl_patterns(sg.LTL_DAC_PATTERNS)
        else:
            raise NotImplementedError()
        self.patterns = [q.relabel(spot.Abc).to_str() for q in gen]


    def add_part(self, formula_so_far, max_size):
        new_pattern: str = self.patterns[random.randint(0, len(self.patterns)-1)]
        used_aps = ''.join(set(re.sub(r'[^a-z]', '', new_pattern)))
        new_aps = self.variables[:] # copy
        random.shuffle(new_aps)
        new_aps = new_aps[:len(used_aps)]
        translation = str.maketrans(used_aps, ''.join(new_aps)) # map existing aps to random new aps
        new_pattern = new_pattern.translate(translation)
        new_pattern_f = ltl_parser.ltl_formula(new_pattern, 'spot')
        new_formula = ltl_parser.F_AND(formula_so_far, new_pattern_f) if formula_so_far else new_pattern_f

        if new_formula.size() > max_size:
            return formula_so_far
        else:
            return self.add_part(new_formula, max_size)

    def __next__(self):
        max_size = random.randint(*self.tree_size)
        return self.add_part(None, max_size)
