import time
import numpy as np
import tensorflow as tf
from data.synthetic.synthetic_binary import gen_data


class BatchGenerator(object):

    def __init__(self, tensor_in, tensor_out, batch_size, seq_length):
        """初始化mini-batch产生器，BaseClass
        Input:
            batch_size: 每一个mini-batch里面有多少样本。
            seq_length: 每一个样本的长度，和batch_size一起决定了每个minibatch的数据量。
        """
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.tensor_in = tensor_in
        self.tensor_out = tensor_out

        self.create_batches()
        self.reset_batch_pointer()

    def reset_batch_pointer(self):
        self.pointer = 0

    def create_batches(self):
        self.num_batches = int(self.tensor_in.size / (self.batch_size * self.seq_length))
        self.tensor_in = self.tensor_in[:self.num_batches * self.batch_size * self.seq_length]
        self.tensor_out = self.tensor_out[:self.num_batches * self.batch_size * self.seq_length]

        # When the data (tesor) is too small, let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.x_batches = np.split(self.tensor_in.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(self.tensor_out.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y


class CopyBatchGenerator(BatchGenerator):

    def __init__(self, data, batch_size, seq_length):
        """初始化mini-batch产生器

        输入一个长度为T的sequence，sequence的前T-1个元素为input，
          sequence的后面T-1个元素为output。用来训练RNNLM。

        Input:
            batch_size: 每一个mini-batch里面有多少样本。
            seq_length: 每一个样本的长度，和batch_size一起决定了每个minibatch的数据量。
        """
        self.batch_size = batch_size
        self.seq_length = seq_length

        tensor_in = np.array(data)
        tensor_out = np.copy(tensor_in)
        tensor_out[:-1] = tensor_in[1:]
        tensor_out[-1] = tensor_in[0]

        super(CopyBatchGenerator, self).__init__(tensor_in, tensor_out, batch_size, seq_length)


class PredBatchGenerator(BatchGenerator):

    def __init__(self, data_in, data_out, batch_size, seq_length):
        """初始化mini-batch产生器

        输入两个长度为T的sequence，其中一个是输入sequence，另一个是输出sequence。

        Input:
            batch_size: 每一个mini-batch里面有多少样本。
            seq_length: 每一个样本的长度，和batch_size一起决定了每个minibatch的数据量。
        """
        self.batch_size = batch_size
        self.seq_length = seq_length

        tensor_in = np.array(data_in)
        tensor_out = np.array(data_out)
        super(PredBatchGenerator, self).__init__(tensor_in, tensor_out, batch_size, seq_length)


class CharRNNLM(object):

    def __init__(self, batch_size, num_unrollings, vocab_size,
                 hidden_size, embedding_size, learning_rate):
        """Character-2-Character RNN 模型。

        这个模型的训练数据是两个相同长度的sequence，其中一个sequence是input，
          另外一个sequence是output。

        """
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.input_data = tf.placeholder(tf.int64, [self.batch_size, self.num_unrollings], name='inputs')
        self.targets = tf.placeholder(tf.int64, [self.batch_size, self.num_unrollings], name='targets')

        cell_fn = tf.nn.rnn_cell.BasicRNNCell

        params = dict()
        # 单元 这里复习一下 单元有状态 输入、输出
        cell = cell_fn(self.hidden_size, **params)

        with tf.name_scope('initial_state'):
            # 初始化细胞状态
            self.zero_state = cell.zero_state(self.batch_size, tf.float32)
            # 将初始状态标记好
            self.initial_state = tf.placeholder(tf.float32, [self.batch_size, cell.state_size], 'initial_state')

        with tf.name_scope('embedding_layer'):
            ## 定义词向量参数， fixme 并通过查询将输入的整数序列每一个元素转换为embedding向量
            # 如果提供了embedding的维度，我们声明一个embedding参数，即词向量参数矩阵
            # 否则，我们使用Identity矩阵作为词向量参数矩阵
            if embedding_size > 0:
                self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            else:
                self.embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

        with tf.name_scope('slice_inputs'):
            # 我们将要使用static_rnn方法，需要将长度为num_unrolling的序列切割成
            # num_unrolling个单位，存在一个list里面,
            # 即，输入格式为：
            # [ num_unrollings, (batch_size, embedding_size)]
            sliced_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(
                axis=1, num_or_size_splits=self.num_unrollings, value=inputs)]

        # 调用static_rnn方法，作forward propagation
        # 为方便阅读，我们将static_rnn的注释贴到这里
        # 输入：
        #     inputs: A length T list of inputs, each a Tensor of shape [batch_size, input_size]
        #     initial_state: An initial state for the RNN.
        #                If cell.state_size is an integer, this must be a Tensor of appropriate
        #                type and shape [batch_size, cell.state_size]
        # 输出：
        #     outputs: a length T list of outputs (one for each input), or a nested tuple of such elements.
        #     state: the final state
        outputs, final_state = tf.nn.static_rnn(
            cell=cell,
            inputs=sliced_inputs,
            initial_state=self.initial_state)
        self.final_state = final_state

        with tf.name_scope('flatten_outputs'):
            flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])

        with tf.name_scope('flatten_targets'):
            flat_targets = tf.reshape(tf.concat(axis=1, values=self.targets), [-1])

        with tf.variable_scope('softmax') as sm_vs:
            softmax_w = tf.get_variable('softmax_w', [hidden_size, vocab_size])
            softmax_b = tf.get_variable('softmax_b', [vocab_size])
            self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)

        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=flat_targets)
            self.mean_loss = tf.reduce_mean(loss)

        with tf.name_scope('loss_montor'):
            count = tf.Variable(1.0, name='count')
            sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')

            self.reset_loss_monitor = tf.group(sum_mean_loss.assign(0.0),
                                               count.assign(0.0), name='reset_loss_monitor')
            self.update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss + self.mean_loss),
                                                count.assign(count + 1), name='update_loss_monitor')

            with tf.control_dependencies([self.update_loss_monitor]):
                self.average_loss = sum_mean_loss / count
                self.ppl = tf.exp(self.average_loss)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.0))

        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.mean_loss, tvars)
        with tf.variable_scope("encoder") as scope:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    # 运行一个epoch
    # 注意我们将session作为一个input argument
    # 参考下图解释
    def run_epoch(self, session, batch_generator, learning_rate, freq=10):
        epoch_size = batch_generator.num_batches

        extra_op = self.train_op

        state = self.zero_state.eval()

        self.reset_loss_monitor.run()
        batch_generator.reset_batch_pointer()
        start_time = time.time()
        for step in range(epoch_size):
            x, y = batch_generator.next_batch()

            ops = [self.average_loss, self.ppl, self.final_state, extra_op, self.global_step]

            feed_dict = {self.input_data: x, self.targets: y,
                         self.initial_state: state,
                         self.learning_rate: learning_rate}

            results = session.run(ops, feed_dict)
            # option 1. 将上一个 minibatch 的 final state
            #   作为下一个 minibatch 的 initial state
            average_loss, ppl, state, _, global_step = results
            # option 2. 总是使用 0-tensor 作为下一个 minibatch 的 initial state
            # average_loss, ppl, final_state, _, global_step = results

        return ppl, global_step


def run():
    batch_size = 16
    # 要生成的训练数据段的数目
    num_unrollings = 20
    vocab_size = 2
    # 隐藏层大小
    hidden_size = 16
    # 输入向量大小
    embedding_size = 16
    # 学习率
    learning_rate = 0.01

    model = CharRNNLM(batch_size, num_unrollings,
                      vocab_size, hidden_size, embedding_size, learning_rate)
    dataset = gen_data(size=1000000)
    batch_size = 16
    seq_length = num_unrollings
    batch_generator = PredBatchGenerator(data_in=dataset[0],
                                         data_out=dataset[1],
                                         batch_size=batch_size,
                                         seq_length=seq_length)
    # batch_generator = BatchGenerator(dataset[0], batch_size, seq_length)
    session = tf.Session()
    with session.as_default():
        for epoch in range(5):
            session.run(tf.global_variables_initializer())
            ppl, global_step = model.run_epoch(session, batch_generator, learning_rate, freq=10)
            print(ppl)

    all_vars = [node.name for node in tf.global_variables()]
    for var in all_vars:
        print(var)
    tf.get_variable_scope().reuse_variables()
    valid_model = CharRNNLM(batch_size, num_unrollings,
                            vocab_size, hidden_size, embedding_size, learning_rate)


if __name__ == '__main__':
    run()
