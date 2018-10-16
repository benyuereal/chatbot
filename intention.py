# 预测意图
import json
import io
import fasttext


# 首先拿到数据

def preview_data():
    # 首先拿到数据
    training_data = 'training_data.json'

    with io.open(training_data, encoding='utf-8-sig') as training_data_file:
        # 加载成json文件
        data = json.loads(training_data_file.read())
    # 标签和文本
    labels, texts = [], []
    # json数据
    #  "rasa_nlu_data": {
    # 		"common_examples": [{
    # 				"text": "hey",
    # 				"intent": "greet",
    # 				"entities": []
    # 			},
    # json 数据的对象保存在 'rasa_nlu_data'.'common_examples'
    for element in data['rasa_nlu_data']['common_examples']:
        # python语言的json数据不和java一样 python是用['key']来取数据
        texts.append(element['text'])
        labels.append('__label__' + element['intent'])
    with open('intent_small_train.txt', 'w') as small_train:
        with open('intent_small_valid.txt', 'w') as small_valid:
            # 从标签里面取到标签值
            for i in range(len(labels)):
                # 这个可以理解为，fixme
                # 如果标签的第一个数据，那么就将标签的[值 意图值]\n 放入到训练的数据中
                # 如果是当前标签值和上一个标签值不一样的话 例如
                # 	affirm : yes
                # 	goodbye : bye
                # 	goodbye : bye
                # goodbye 就会被放入训练集合
                # 这个就保证了 每一个标签都会至少被放入测试集合一次
                if i == 0 or labels[i] != labels[i - 1]:
                    small_valid.write(labels[i] + ' ' + texts[i] + '\n')
                else:
                    small_train.write(labels[i] + ' ' + texts[i] + '\n')

    print('所有的intention和text')
    print(set([x[9:] for x in labels]))
    print('\n')
    print('所有的 (intent, text) 样本:')
    xs = sorted([(labels[i], texts[i]) for i in range(len(labels))])

    for i in range(len(labels)):
        print('\t%s : %s' % (xs[i][0][9:], xs[i][1]))


# 这里是训练集合
def train():
    # 训练的学习率
    global params_train, params_valid
    learning_rates = [0.01, 0.05, 0.002]
    # 训练批次
    dims = [5, 10, 25, 50, 75, 100]
    # 定义最佳的训练和测试结果
    best_train, best_valid = 0, 0
    for learning_rate in learning_rates:
        for dim in dims:
            # 进行文本分类，可以看成一个word2vector word2vector最后输出的是可能的中间次 然后fasttext最后的输出是标签
            classifier = fasttext.supervised(input_file='intent_small_train.txt',
                                             output='intent_model',
                                             #   切割值是__label__
                                             label_prefix='__label__',
                                             dim=dim,
                                             lr=learning_rate,
                                             epoch=50)
            result_train = classifier.test('intent_small_train.txt')
            result_valid = classifier.test('intent_small_valid.txt')
            if result_train.precision > best_train:
                best_train = result_train.precision
                params_train = (learning_rate, dim, result_train)

            if result_valid.precision > best_valid:
                best_valid = result_valid.precision
                params_valid = (learning_rate, dim, result_valid)

    print(best_valid)
    print(params_train)
    print(best_valid)
    print(params_valid)
    # 最后利用最佳的学习率和dim来训练
    classifier = fasttext.supervised(input_file='intent_small_train.txt',
                                     output='intent_model',
                                     label_prefix='__label__',
                                     dim=params_valid[1],
                                     lr=params_valid[0],
                                     epoch=50)
    # 这是预测
    print(classifier.predict(['ok ', 'hello', 'bye bye', 'show me chinese restaurants'], k=1))



def main():
    preview_data()
    train()


if __name__ == '__main__':
    main()
