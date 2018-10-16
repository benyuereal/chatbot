# 首先 引入包
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 设置数据的大小
total_size = 1000000


def generate_data(size=total_size):
    """ 按照上图生成合成序列数据

        Arguments:
            size: input 和 output 序列的总长度

        Returns:
            X, Y: input 和 output 序列，rank-1的numpy array （即，vector)
        """
    # 可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回。
    # 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
    # a1 = np.random.choice(a=5, size=3, replace=False, p=None)

    # fixme 产生随机采样 在每个位置上 随机产生0或者1
    x = np.array(np.random.choice(2, size=(size,)))

    y = []
    for i in range(size):
        # 基础概率 产生1的概率是0.5
        threshold = 0.5
        # 假如目标位置往前数三个 那个位置的数字是1 那么目标位置的数字是1的概率增加0.5
        if x[i - 3] == 1:
            threshold += 0.5
        # 如果目标位置往前面数八个，那个位置的数字是1 那么目标数字是1的概率就减少0.25
        # 总之 目标位置的数字出现1的概率总是受到前面位置数字值的影响
        if x[i - 8] == 1:
            threshold -= 0.25
        # 返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
        # 总的来看 np.random.rand()有百分之50的概率比 0.5要大，有百分之50的概率比0.5小
        if np.random.rand() > threshold:
            y.append(0)
        else:
            y.append(1)
    return x, np.array(y)

    # 下面实现一个简单的rnn


def gen_batch(raw_data, batch_size, num_steps):
    """产生minibatch数据

    Arguments:
        raw_data: 所有的数据， (input, output) tuple
        batch_size: 一个minibatch包含的样本数量；每个样本是一个sequence
        num_step: 每个sequence样本的长度

    Returns:
        一个generator，在一个tuple里面包含一个minibatch的输入，输出序列
    """

    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


# 状态长度
state_size = 16
# 设置批次数
batch_size = 32
# 设置步长
num_steps = 4
# 将输入序列中的每一个0,1数字转化为二维one-hot向量
num_classes = 2


def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(generate_data(), batch_size, num_steps)


def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    #     tf.matmul是矩阵乘法
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)


def rnn():
    # 函数用于清除默认图形堆栈并重置全局默认图形
    tf.reset_default_graph()

    # 这里等于说用tf定义一个矩阵 矩阵的大小是(32,4),名字是 input_placeholder
    x = tf.placeholder(tf.int32, [batch_size, num_steps],
                       name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps],
                       name='labels_placeholder')

    """
    RNN Inputs， 将前面定义的placeholder输入到RNN cells
    """
    # num_classes 表示输出的长度
    x_one_hot = tf.one_hot(x, num_classes)
    # stack是拼接 unstack是分解 axis=是按照输入向量的几号位进行分解
    rnn_inputs = tf.unstack(x_one_hot, axis=1)  # [ num_steps, [batch_size, num_classes]]

    # 设置w、b
    with tf.variable_scope('rnn_cell'):
        W = tf.get_variable('W', [num_classes + state_size, state_size],
                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        b = tf.get_variable('b', [state_size],
                            initializer=tf.constant_initializer(0.0))
    # init_state, state 和 final_state:
    rnn_outputs = []
    init_state = tf.zeros([batch_size, state_size])
    state = init_state
    for rnn_input in rnn_inputs:
        # 将输入转换成rnn_cell
        state = rnn_cell(rnn_input, state)
        rnn_outputs.append(state)
    final_state = rnn_outputs[-1]

    """
    计算损失函数，定义优化器
    """
    # 从每个 time frame 的 hidden state
    # 映射到每个 time frame 的最终 output（prediction）；
    # 和CBOW或者SKIP-GRAM的最上一层相同

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    predictions = [tf.nn.softmax(logit) for logit in logits]

    # 计算损失函数
    # 计算损失函数
    y_as_list = tf.unstack(y, num=num_steps, axis=1)
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for
              logit, label in zip(logits, y_as_list)]
    total_loss = tf.reduce_mean(losses)

    # 定义优化器
    learning_rate = 0.1
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
    # 上面有一个tf.Variable的例子
    # 在tf.variable_scope('rnn_cell')和tf.variable_scope('softmax')中，各自有两个用W和b表示的tf.Variable. 因为在不同的variable_scope，即便使用同样的名字，表示的是不同的对象
    # 打印graph的nodes：
    all_node_names = [node for node in tf.get_default_graph().as_graph_def().node]
    # 或者：
    # all_node_names = [node for node in tf.get_default_graph().get_operations()]
    all_node_values = [node.values() for node in tf.get_default_graph().get_operations()]

    for i in range(0, len(all_node_values), 50):
        print('output and operation %d:' % i)
        print(all_node_values[i])
        print('-------------------')
        print(all_node_names[i])
        print('\n')
        print('\n')

    for i in range(len(all_node_values)):
        print('%d: %s' % (i, all_node_values[i]))

    """
    训练模型的参数
    """

    num_epochs = 4
    verbose = True
    # 画重点：
    # “细节是魔鬼”， 为什么把上一个minibatch的输出，training_state，作为下一个minibatch的输入， init_state:trining_state?
    # for step, (X, Y) in enumerate(epoch):
    #     tr_losses, training_loss_, training_state, _ = sess.run(
    #         [losses, total_loss, final_state, train_step],
    #         feed_dict={x:X, y:Y, init_state:training_state})
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        training_losses = []
        # list1 = ["这", "是", "一个", "测试"]
        # for index, item in enumerate(list1):
        #     print index, item
        # >>>
        # 0 这
        # 1 是
        # 2 一个
        # 3 测试
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = sess.run(
                    [losses, total_loss, final_state, train_step],
                    feed_dict={x: X, y: Y, init_state: training_state})
                training_loss += training_loss_
                if step % 500 == 0 and step > 0:
                    if verbose:
                        print("At step %d, average loss of last 500 steps are %f\n"
                              % (step, training_loss / 500.0))
                    training_losses.append(training_loss / 500.0)
                    training_loss = 0
                    plt.plot(training_losses)  # when num_len = 4, state_size = 16
                    plt.show()


if __name__ == '__main__':
    rnn()
