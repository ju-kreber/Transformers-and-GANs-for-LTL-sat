# implementation based on DeepLTL https://github.com/reactive-systems/deepltl

import sys
import importlib
import os.path as path
import subprocess
import re
from timeit import default_timer as timer
from typing import Tuple

import multiexit

from dlsgs.utils import ltl_parser

DEFAULT_BINARY_PATH = "bin"


def solve_ltl(formula_obj, tool, worker, timeout=None, simplify=False, no_disjunctions=False, witness=True, binary_path=DEFAULT_BINARY_PATH):
    if tool == 'spot':
        formula_str = formula_obj.to_str('spot', spacing='all ops', full_parens=True) # unambiguous form
        finished, res = worker.call(_solve_spot, (formula_str, simplify), timeout)
        if not finished:
            return None, None, {}  # timeout
        sat, trace, d = res
        assert sat is not None
        if sat:
            trace = ltl_parser.ltl_trace(trace, 'spot')
            if no_disjunctions:
                trace.decide_disjunctions()
        else:
            trace = None
        return sat, trace, d
    elif tool == 'aalta':
        formula_str = formula_obj.rewrite(ltl_parser.Token.WEAKUNTIL).to_str('spot', spacing='all ops', full_parens=True) # unambiguous form
        assert not simplify
        if witness:
            sat, trace = aalta_sat_with_evidence(formula_str, timeout=timeout, binary_path=binary_path)
            return sat, ltl_parser.ltl_trace(trace, 'spot') if sat else None, {}
        else:
            try:
                sat = aalta_sat(formula_str, timeout=timeout)
            except RuntimeError:
                sat = None
            return sat, None, {}
    elif tool == 'leviathan':
        formula_str = formula_obj.to_str('spot', spacing='all ops', full_parens=True) # unambiguous form
        assert not simplify
        worker.register_close_function(_close_leviathan)
        finished, res = worker.call(_solve_leviathan, (formula_str, binary_path), timeout)
        if not finished:
            return None, None, {} # timeout
        sat, trace, d = res
        return sat, ltl_parser.ltl_trace(trace, 'spot', symbolic=False) if sat else None, d
    else:
        raise ValueError("Unknown tool")


def _solve_leviathan(formula_str, binary_path=DEFAULT_BINARY_PATH):
    if not 'proc' in globals():
        global proc
        lev_path = path.join(binary_path, 'leviathan')
        proc = subprocess.Popen([lev_path, '-m', '-p', '-v', '4', '/dev/stdin'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)            
        multiexit.register(proc.kill)

    t_start = timer()
    #print(formula_str)
    sys.stdout.flush()
    proc.stdin.write(formula_str + '\n')
    stuff = ''
    while True:
        lastline = proc.stdout.readline()
        if lastline.startswith('SAT;') or lastline.startswith('UNSAT'):
            break
        else:
            stuff += lastline
    d = {'solve_time' : (timer() - t_start) * 1000 }
    d.update(find_int_in_output(r'Found (\d+) eventualities\n', stuff, 'lev_evtls', None))
    d.update(find_int_in_output(r'Total frames: (\d+)\n', stuff, 'lev_frames', None))
    d.update(find_int_in_output(r'Total STEPs: (\d+)\n', stuff, 'lev_steps', None))
    d.update(find_int_in_output(r'Maximum model size: (\d+)\n', stuff, 'lev_max_model_size', None))
    d.update(find_int_in_output(r'Maximum depth: (\d+)\n', stuff, 'lev_max_depth', None))

    if lastline.startswith('SAT;'):
        sat = True
        lastline = lastline[4:]
        parts = lastline.strip().split(' -> ')
        if len(parts) == 1:
            assert parts[0] == '{}'
            trace = 'cycle{1}'
        else:
            loop_to = re.match(r'#(\d+)', parts[-1])
            assert loop_to
            loop_to = int(loop_to.groups()[0])
            parts_ = []
            for p in parts[:-1]:
                assert (p.startswith('{') and p.endswith('}'))
                p = p[1:-1]
                if p:
                    parts_.append(p)
                else:
                    parts_.append('1')
            prefix = '; '.join(parts_[:loop_to])
            cycle =  '{' + '; '.join(parts_[loop_to:]) + '}'
            trace = prefix + ('; ' if prefix else '') + 'cycle' + cycle
    else: # unsat
        sat = False
        trace = None
    return sat, trace, d


def find_int_in_output(regex, output, name, default):
    m = re.search(regex, output)
    if m:
        return {name : int(m.groups()[0])}
    elif default is not None:
        return {name : default}
    else:
        return {}


def _close_leviathan():
    if 'proc' in globals():
        stdout, _ = proc.communicate(timeout=1)
        if stdout != '':
            print('Leviathan had trailing output:', stdout)
        proc.kill()

def _solve_spot(formula_str, simplify):
    if not 'spot' in globals():
        global spot
        spot = importlib.import_module('spot')
    t_start = timer()
    spot_formula = spot.formula(formula_str)
    automaton = spot_formula.translate()
    automaton.merge_edges()
    acc_run = automaton.accepting_run()
    d = {'solve_time' : (timer() - t_start) * 1000 }
    if acc_run is None:
        return False, None, d
    else:
        trace = spot.twa_word(acc_run)
        if simplify:
            trace.simplify()
        return True, str(trace), d


def aalta_sat_with_evidence(formula: str, timeout=None, binary_path=DEFAULT_BINARY_PATH) -> Tuple[bool, str]:
    """Calls aalta to check if the provided formula is satisfiable and returns a witness if so"""
    full_aalta_path = path.join(binary_path, 'aalta')
    try:
        # arguments -l and -c do not seem to be necessary, but include them, for future versions
        res = subprocess.run([full_aalta_path, '-l', '-c', '-e'], input=formula, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True, universal_newlines=True)
    except subprocess.TimeoutExpired:
        print('aalta timed out after {:.2f}s'.format(timeout))
        return None, None
    except subprocess.CalledProcessError as e:
        raise RuntimeError("aalta threw an error: " + e.stderr)
    m = re.fullmatch('please input the formula:\n((?:un)?sat)\n(.*)', res.stdout, re.MULTILINE | re.DOTALL)
    if not m:
        raise RuntimeError("Regular expression for aalta output did not match. Output: '" + res.stdout + "'")
    res_sat, res_trace = m.groups()
    if res_sat == 'unsat':
        return False, None

    # convert aalta trace to spot trace
    assert res_sat == 'sat'
    m = re.fullmatch('(.*[(]\n.*\n[)])\\^w\n', res_trace, re.MULTILINE | re.DOTALL) # not really necessary, more as check
    if not m:
        raise RuntimeError("Regular expression for aalta trace did not match. Trace output: '" + res_trace + "'")
    trace_str = m.groups()[0]
    trace_str = re.sub('[{][}]\n', '1; ', trace_str)                                 # special case, yaay! convert {} directly to 1;
    trace_str = trace_str.replace('{', '').replace(',}', '').replace(',', ' & ')    # convert set {a, !b, c} to formula a & !b & c
    trace_str = trace_str.replace('true', '1')                                      # convert true literal to 1
    trace_str = re.sub('[(]\n', 'cycle{', trace_str)                                # convert ( ... ) to cycle{...}
    trace_str = re.sub('\n[)]$', '}', trace_str)
    trace_str = re.sub('\n', '; ', trace_str)                                       # convert newlines to ;
    return True, trace_str, {}

def aalta_sat(formula: str, timeout=None, binary_path=DEFAULT_BINARY_PATH, mute=False) -> bool:
    """Calls aalta to check if the provided formula is satisfiable"""
    full_aalta_path = path.join(binary_path, 'aalta')
    try:
        res = subprocess.run([full_aalta_path], input=formula, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True, universal_newlines=True)
    except subprocess.TimeoutExpired:
        # print('aalta timed out after {:.2f}s'.format(timeout))
        return None
    except subprocess.CalledProcessError as e:
        if not mute:
            print("!! aalta threw an error: " + e.stderr)
            print('for input formula', formula)
        return None
    m = re.fullmatch('please input the formula:\n((?:un)?sat)\n', res.stdout, re.MULTILINE | re.DOTALL)
    if not m:
        raise RuntimeError("Regular expression for aalta output did not match. Output: '" + res.stdout + "'")
    res_sat = m.groups()[0]
    assert res_sat == 'sat' or res_sat == 'unsat'
    return res_sat == 'sat'
