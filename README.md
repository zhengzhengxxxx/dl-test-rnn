此项目是练习RNN时，实践的一个机器翻译项目，不可用于生产环境

1.create_word.py
将origin_data中的训练文件，按照出现的频率拆词，按照频率由大到小排列，写入target_data文件夹中

2.create_train.py
以create_word.py中生成的字典，将文章翻译成数字id,写入到seq2seq_data文件夹

3.seq2seq_train.py
词典训练文件，首先将整个文章（由id组成）截断成batch形式，每个batch中有若干中英对照的句子。由两个rnn组成，
翻译模型由左右两侧的rnn组成，左侧rnn以0位初始状态，循环读入句子中的每个单词，最后输出状态向量ht。右侧rnn
以ht作为初始状态输入，第一个神经元固定读入<sos>，输入为预测的词语，以后的每个神经元输入x,都是上一个神经元
的输出y。通过计算输出y和实际label的交叉熵，最终拟合函数，并保存

4.seq2seq_test.py
载入上一步的保存文件，左侧rnn与seq2seq_train.py中的编码部分一致，右侧rnn（解码）则稍有不同，解码每次输出
一个最大的估计值作为输出
