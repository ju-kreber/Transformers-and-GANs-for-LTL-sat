# implementation based on DeepLTL https://github.com/reactive-systems/deepltl

# pylint: disable = line-too-long, no-member
import sys, os
import json
import math
from argparse import Namespace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # reduce TF verbosity

import tensorflow as tf
from tensorflow.io import gfile #pylint: disable=import-error

from dlsgs.train.common import *
from dlsgs.utils import utils
from dlsgs.transformer import lr_schedules
from dlsgs.transformer.base import Transformer
from dlsgs.transformer.derived import EncoderOnlyTransformer
from dlsgs.universal_transformer.derived import UniversalEncoderOnlyTransformer
from dlsgs.transformer.common import create_model
from dlsgs.utils import vocabulary
from dlsgs.utils import datasets
from dlsgs.universal_transformer.utils import StoppedPercentMetric, ProxyCallback, DetailedTFCallback


def run():
    # Argument parsing
    parser = argparser()
    # add specific arguments
    parser.add_argument('--problem', type=str, default='ltl', help='available problems: ltl, prop')
    parser.add_argument('--d-embed-enc', type=int, default=128)
    parser.add_argument('--d-embed-dec', type=int, default=None)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--ff-activation', default='relu')
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=4000)
    parser.add_argument('--tree-pe', action='store_true', default=False, help='enable tree positional encoding, default false')
    parser.add_argument('--format', type=str, default=None, help='format of formulas, needs to be specified if tree positional encoding is used')
    parser.add_argument('--force-load', default=False, action='store_true', help='Assure that weigths from checkpoint are loaded, fail otherwise')
    parser.add_argument('--implementation', default='trans')
    parser.add_argument('--max-encode-length', type=int, help='filter input length')
    parser.add_argument('--load-from', type=str)
    witness = parser.add_mutually_exclusive_group()
    witness.add_argument('--witness', dest='witness', action='store_true', default=False, help='output a witness for satisfiable instances, default False')
    witness.add_argument('--no-witness', dest='witness', action='store_false', default=False, help='output no wintess for satisfiable instances, default.')
    parser.add_argument('--only-sat', action='store_true', default=False, help='only consider satisfiable instances, default False')
    params = parser.parse_args()
    params_dict = vars(params)

    # Parameters
    params_dict = setup(**params_dict)
    impl = params_dict['implementation']
    if impl.startswith('trans'):
        model_class = Transformer
    elif impl.startswith('enc'):
        model_class = EncoderOnlyTransformer

    d = {}
    if 'uni' in impl:
        d['ut_base_layers'] = 1
        d['ut_pre_layers'] = 0
        d['ut_post_layers'] = 0
        d['ut_max_iterations'] = 10
        d['ut_random_all_stop'] = 0.00
        d['ut_stop_base'] = 0.0
        d['ut_stop_map_certainty'] = True
        d['ut_iterations'] = 'max'
        d['ut_gradient_method'] = 'checkpoint' #'tape' # tape, checkpoint, tape+checkpoint
        d['ut_test_iterations'] = []
    if 'enc' in impl:
        del params_dict['d_embed_dec']
        d['enc_accumulation'] = 'first' # first, mean-before, mean-after
    if 'uni' in impl and 'enc' in impl:
        del params_dict['num_layers']
    if not params.test:
        for key in ['alpha', 'beam_size', 'eval_name', 'no_auto']:
            del params_dict[key]
    else:
        for key in ['warmup_steps', 'epochs', 'initial_epoch']:
            del params_dict[key]
    for k, v in d.items(): # update only non-existent values
        if k not in params_dict:
            params_dict[k] = v
    params = Namespace(**params_dict)


    # Dataset specification
    dataset_name = params.ds_name
    with gfile.GFile(os.path.join(params.data_dir, params.ds_name, 'info.json'), 'r') as f:
        ds_info = json.loads(f.read())
    assert params.problem == ds_info['problem']
    aps = list(map(chr, range(97, 97 + ds_info['num_variables'])))

    consts = ['0', '1']
    if params.problem == 'ltl':
        input_vocab = vocabulary.LTLVocabulary(aps=aps, consts=consts + ['%'], ops=['U', 'X', '!', '&', 'F', 'G', 'W', '|', '>', '='], eos=not params.tree_pe)
        if 'witness' in ds_info['subproblem'] and params.witness:
            target_vocab = vocabulary.TraceVocabulary(aps=aps, consts=consts, ops=['&', '!', '|'])
            print('This is a variable-text output problem.')
        else:
            target_vocab = vocabulary.TraceVocabulary(aps=[], consts=consts, ops=[], special=[], start=True, eos=False, pad=True)
            print('This is a binary decision problem.')
        dataset = datasets.LTLTracesDataset(dataset_name, input_vocab, target_vocab, data_dir=params.data_dir, reduce_witness_to_sat=not params.witness, only_sat=params.only_sat)
        additional_in_tokens = int(not params.tree_pe)
        additional_out_tokens = int(params.witness)
        max_encode_length = ds_info['max_in_length'] + additional_in_tokens
        max_decode_length = (ds_info['max_out_length'] if params.witness else 1) + additional_out_tokens
        if params.max_encode_length is None:
            base_2_enc = math.ceil(math.log(max_encode_length, 2))
            params.max_encode_length = 2**base_2_enc if max_encode_length / 2**base_2_enc >= 0.8 else max_encode_length # use next power of 2
        base_2_dec = math.ceil(math.log(max_decode_length, 2))
        params.max_decode_length = 2**base_2_dec if max_decode_length / 2**base_2_dec >= 0.8 else max_decode_length # use next power of 2
        #params.max_decode_length = 64
    elif params.problem == 'prop':
        input_vocab = vocabulary.LTLVocabulary(aps, consts, ['!', '&', '|', '>'], eos=not params.tree_pe)
        target_vocab = vocabulary.TraceVocabulary(aps, consts, [], special=[])
        dataset = datasets.BooleanSatDataset(dataset_name, data_dir=params.data_dir, formula_vocab=input_vocab, assignment_vocab=target_vocab)
        params.max_encode_length = ds_info['max_in_length'] + 1
        params.max_decode_length = ds_info['max_out_length'] + 1
    else:
        print(f'{params.problem} is not a valid problem\n')
        return
    params.input_vocab_size = input_vocab.vocab_size()
    params.input_pad_id = input_vocab.pad_id
    params.target_vocab_size = target_vocab.vocab_size()
    params.target_start_id = target_vocab.start_id
    params.target_eos_id = target_vocab.eos_id
    params.target_pad_id = target_vocab.pad_id
    params.dtype = 'float32'

    print('Specified dimension of encoder embedding:', params.d_embed_enc)
    params.d_embed_enc -= params.d_embed_enc % params.num_heads  # round down
    try:
        if params.d_embed_dec is None:
            params.d_embed_dec = params.d_embed_enc
        print('Specified dimension of decoder embedding:', params.d_embed_dec)
        params.d_embed_dec -= params.d_embed_dec % params.num_heads  # round down
    except AttributeError:
        pass
    print('Parameters:')
    params_dict = vars(params)
    for key, val in params_dict.items():
        print('{:25} : {}'.format(key, val))


    # Get datasets
    if not params.test:  # train mode
        if params.problem == 'ltl':
            train_dataset, val_dataset = dataset.get_dataset(['train', 'val'], max_length_formula=params.max_encode_length-additional_in_tokens,
               max_length_trace=params.max_decode_length-additional_out_tokens, prepend_start_token=False, tree_pos_enc=params.tree_pe)
        if params.problem == 'prop':
            train_dataset, val_dataset = dataset.get_dataset(splits=['train', 'val'], tree_pos_enc=params.tree_pe)
        if params.samples is not None:
            train_dataset = train_dataset.take(int(params.samples))
            val_dataset = val_dataset.take(int(params.samples/10))
        train_dataset = datasets.prepare_dataset_no_tf(train_dataset, params.batch_size, params.d_embed_enc, shuffle=True, pos_enc=params.tree_pe)
        val_dataset = datasets.prepare_dataset_no_tf(val_dataset, params.batch_size, params.d_embed_enc, shuffle=False, pos_enc=params.tree_pe)
    else:  # test mode
        if params.problem == 'ltl':
            test_dataset, = dataset.get_dataset(['test'], max_length_formula=params.max_encode_length - additional_in_tokens,
                max_length_trace=params.max_decode_length - additional_out_tokens, prepend_start_token=False, tree_pos_enc=params.tree_pe)
        if params.problem == 'prop':
            test_dataset, = dataset.get_dataset(splits=['test'], tree_pos_enc=params.tree_pe)
        if params.samples is not None:
            test_dataset = test_dataset.take(int(params.samples))


    # Model & Training specification
    if not params.test:  # train mode
        learning_rate = lr_schedules.TransformerSchedule(params.d_embed_enc, warmup_steps=params.warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        callbacks = [checkpoint_callback(save_weights_only=True, **params_dict)]
    else:
        optimizer = tf.keras.optimizers.Adam() #todo naja..
        callbacks = []

    if impl.startswith('univenc'):
        if params.witness:
            raise ValueError('Cannot run this model in witness mode')
        model = UniversalEncoderOnlyTransformer(params_dict, training=True)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), StoppedPercentMetric()]
        for iterations in params.ut_test_iterations:
            metrics.append(tf.keras.metrics.SparseCategoricalAccuracy(name='SCA_at_{}_it'.format(iterations)))
            # TODO: doc: must use compiled metrics in training loop, but no meaning here.
        run_eagerly = True
        if not params.test:
            callbacks.extend([ProxyCallback(), DetailedTFCallback(get_log_dir(**params_dict), log_target_iterations=True)])
    elif impl.startswith('enc') or impl.startswith('trans'):
        if impl.startswith('enc') and params.witness:
            raise ValueError('Cannot run this model in witness mode')
        model_inputs = {'indata' : tf.int32}
        if params.tree_pe:
            model_inputs['positional_encoding'] = tf.float32
        if not params.test or not params.witness:
            model_inputs['targets'] = tf.int32
            model_outputs = ['predictions']    
        else:
            model_outputs = ['decodings']
        model = create_model(model_class, params_dict, inputs=model_inputs, outputs=model_outputs, training=not params.test)
        loss, metrics = None, None
        run_eagerly = False
        if not params.test:
            callbacks.append(tf.keras.callbacks.TensorBoard(get_log_dir(**params_dict)))
    else:
        raise ValueError('Unknown implementation')    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)


    # Load from save
    if params.test:
        params.force_load = True
    latest_checkpoint = last_checkpoint(**params_dict)
    if latest_checkpoint:
        model.load_weights(latest_checkpoint).expect_partial()
        print(f'Loaded weights from checkpoint {latest_checkpoint}')
    elif params.force_load:
        sys.exit('Failed to load weights, no checkpoint found!')
    else:
        print('No checkpoint found, creating fresh parameters')
    sys.stdout.flush()
    log_params(**params_dict)
    sys.stdout.flush()


    # Do it!
    if not params.test: # Train!
        model.fit(train_dataset, epochs=params.epochs, validation_data=val_dataset, validation_freq=1, callbacks=callbacks, initial_epoch=params.initial_epoch, verbose=2, shuffle=False)
    else: # Test!
        if not params.witness:
            test_dataset = datasets.prepare_dataset_no_tf(test_dataset, params.batch_size, params.d_embed_enc, shuffle=False, pos_enc=params.tree_pe)
            model.evaluate(test_dataset, verbose=2)
        else:
            padded_shapes = ([None], [None]) if not params.tree_pe else ([None], [None, params.d_embed_enc], [None])
            test_dataset = test_dataset.padded_batch(params.batch_size, padded_shapes=padded_shapes)
            if params.problem == 'ltl':
                test_and_analyze_ltl(model, test_dataset, input_vocab, target_vocab, plot_name=params.eval_name, log_name=params.eval_name, **params_dict)
            elif params.problem == 'prop':
                test_and_analyze_sat(model, test_dataset, input_vocab, target_vocab, log_name=params.eval_name, **params_dict)


if __name__ == '__main__':
    run()
