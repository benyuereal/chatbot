import argparse
import numpy as np


def config_train(args=''):
    parser = argparse.ArgumentParser()

    # hyper-parameters to configure the datasets.
    # 数据相关的超参数
    data_args = parser.add_argument_group('Dataset Options')
    data_args.add_argument('--data_file', type=str,
                           default='data/ice_and_fire_zh/ice_and_fire_utf8.txt',
                           help='data file')
    data_args.add_argument('--encoding', type=str, default='utf-8',
                           help='the encoding format of data file.')
    data_args.add_argument('--num_unrollings', type=int, default=20,
                           help='number of unrolling steps.')
    data_args.add_argument('--train_frac', type=float, default=0.9,
                           help='fraction of data used for training.')
    data_args.add_argument('--valid_frac', type=float, default=0.05,
                           help='fraction of data used for validation.')
    # test_frac is computed as (1 - train_frac - valid_frac).

    # hyper-parameters to configure the neural network.
    # 模型结构相关的超参数
    network_args = parser.add_argument_group('Model Arch Options')
    network_args.add_argument('--embedding_size', type=int, default=128,
                              help='size of character embeddings, 0 for one-hot')
    network_args.add_argument('--hidden_size', type=int, default=256,
                              help='size of RNN hidden state vector')
    network_args.add_argument('--cell_type', type=str, default='lstm',
                              help='which RNN cell to use (rnn, lstm or gru).')
    network_args.add_argument('--num_layers', type=int, default=2,
                              help='number of layers in the RNN')

    # hyper-parameters to control the training.
    # 训练和优化相关的超参数
    training_args = parser.add_argument_group('Model Training Options')
    # 1. Parameters for iterating through samples
    training_args.add_argument('--num_epochs', type=int, default=50,
                               help='number of epochs')
    training_args.add_argument('--batch_size', type=int, default=20,
                               help='minibatch size')
    # 2. Parameters for dropout setting.
    training_args.add_argument('--dropout', type=float, default=0.0,
                               help='dropout rate, default to 0 (no dropout).')
    training_args.add_argument('--input_dropout', type=float, default=0.0,
                               help=('dropout rate on input layer, default to 0 (no dropout),'
                                     'and no dropout if using one-hot representation.'))
    # 3. Parameters for gradient descent.
    training_args.add_argument('--max_grad_norm', type=float, default=5.,
                               help='clip global grad norm')
    training_args.add_argument('--learning_rate', type=float, default=5e-3,
                               help='initial learning rate')

    # Parameters for manipulating logging and saving models.
    # 学习日志和结果相关的超参数
    logging_args = parser.add_argument_group('Logging Options')
    # 1. Directory to output models and other records.
    logging_args.add_argument('--output_dir', type=str,
                              default='demo_model',
                              help=('directory to store final and'
                                    ' intermediate results and models'))
    # 2. Parameters for printing messages.
    logging_args.add_argument('--progress_freq', type=int, default=100,
                              help=('frequency for progress report in training and evalution.'))
    logging_args.add_argument('--verbose', type=int, default=0,
                              help=('whether to show progress report in training and evalution.'))
    logging_args.add_argument('--debug', dest='debug', action='store_true',
                              help='show debug information')
    logging_args.add_argument('--test', dest='test', action='store_true',
                              help=('parameter for unittesting. Use the first 1000 '
                                    'character to as data to test the implementation'))
    # 3. Parameters to feed in the initial model and current best model.
    logging_args.add_argument('--init_model', type=str,
                              default='', help=('initial model'))
    logging_args.add_argument('--best_model', type=str,
                              default='', help=('current best model'))
    logging_args.add_argument('--best_valid_ppl', type=float,
                              default=np.Inf, help=('current valid perplexity'))
    # 4. Parameters for using saved best models.
    logging_args.add_argument('--init_dir', type=str, default='',
                              help='continue from the outputs in the given directory')

    args = parser.parse_args(args.split())

    return args


def config_sample(args=''):
    parser = argparse.ArgumentParser()

    # hyper-parameters for using saved best models.
    # 学习日志和结果相关的超参数
    logging_args = parser.add_argument_group('Logging_Options')
    logging_args.add_argument('--init_dir', type=str,
                              default='demo_model/',
                              help='continue from the outputs in the given directory')

    # hyper-parameters for sampling.
    # 设置sampling相关的超参数
    testing_args = parser.add_argument_group('Sampling Options')
    testing_args.add_argument('--max_prob', dest='max_prob', action='store_true',
                              help='always pick the most probable next character in sampling')
    testing_args.set_defaults(max_prob=False)

    testing_args.add_argument('--start_text', type=str,
                              default='The meaning of life is ',
                              help='the text to start with')

    testing_args.add_argument('--length', type=int,
                              default=100,
                              help='length of sampled sequence')

    testing_args.add_argument('--seed', type=int,
                              default=-1,
                              help=('seed for sampling to replicate results, '
                                    'an integer between 0 and 4294967295.'))

    args = parser.parse_args(args.split())

    return args