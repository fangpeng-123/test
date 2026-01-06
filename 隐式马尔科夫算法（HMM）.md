### 原理分析及算法流程

1. 首先用标准的13维MFCC加上一阶和二阶导数训练单音素GMM系统，采用倒谱均值归一化（CMN）来降低通道效应。然后基于具有由LDA和MLLT变换的特征的单音系统构造三音GMM系统，最后的GMM系统用于为随后的DNN训练生成状态对齐。

2. 基于GMM系统提供的对齐来训练DNN系统，特征是40维FBank，并且相邻的帧由11帧窗口（每侧5个窗口）连接。连接的特征被LDA转换，其中维度降低到200。然后应用全局均值和方差归一化以获得DNN输入。DNN架构由4个隐藏层组成，每个层由1200个单元组成，输出层由3386个单元组成。 基线DNN模型用交叉熵的标准训练。 使用随机梯度下降（SGD）算法来执行优化。 将迷你批量大小设定为256，初始学习率设定为0.008。

3. 被噪声干扰的语音可以使用基于深度自动编码器（DAE）的噪声消除方法。DAE是自动编码器（AE）的一种特殊实现，通过在模型训练中对输入特征引入随机破坏。已经表明，该模型学习低维度特征的能力非常强大，并且可以用于恢复被噪声破坏的信号。在实践中，DAE被用作前端管道的特定组件。输入是11维Fbank特征（在均值归一化之后），输出是对应于中心帧的噪声消除特征。然后对输出进行LDA变换，提取全局标准化的常规Fbank特征，然后送到DNN声学模型（用纯净语音进行训练）。

   [算法流程]: https://blog.csdn.net/snowdroptulip/article/details/78943748

##### 关键字解释

- MFCC：一种能够模仿人类听觉感知特性，被广泛应用于语音处理的特征提取方法，最终都得到的特征常量是13维。
- CMN：是一种常用的特征标准化技术，旨在消除因录音设备或环境差异导致的通道效应。具体做法是对所有utterance的MFCC特征进行全局均值归一化，即减去整个语料库中所有帧MFCC特征的平均值。这样做的目的是让不同录音条件下的特征具有相似的统计特性，从而提高模型的鲁棒性和泛化能力。
- SGD：随机梯度下降算法，是一种用于优化机器学习模型参数的迭代方法。它是梯度下降算法的一种变体，广泛应用于训练深度学习模型和大规模数据集中的其他机器学习模型。

### 使用kaldi包安装编译

简介：kaldi是一种n-gram模型，语言模型的作用是将语言装换为图这汇总数据结构，高效的语音模型本质是从语音对应的图的网络中寻找最短路径。

1. 安装kaldi包：

```shell
sudo git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin golden
```

2. 检查需要安装的库：进入tools文件下运行extras/check_dependencies.sh文件

```shell
cd kaldi/tools
extras/check_dependencies.sh ##这可以检查出需要安装的库，直接根据提示进行安装
```

3. 在./tool目录中输入

```shell
make      ## 或者执行nproc 命令查看有多少个核可以用于并行编译加快速度，例如本机带有8核则执行
make -j 8
```

4. 切换到./src目录下，运行如下命令进行编译

```shell
./configure --share
make depend
make -j  4 ## 或者执行make
```

全部完成且终端不报错误，显示done时表明编译成功。



### kaldi包内脚本训练模型

1. 数据集准备：第一种直接运行脚本，脚本中写好了如何和获取数据集以及数据集保存解压的位置见./esg/aishell/s5/.run.sh，但此种方法可能存在文件写入失败的保错建议使用第二种官网下载数据集再进行训练。

   ```shell
   data=./data			#想要写入的文件位置，一般写为./data安全性高
   data=/media/yls/1T硬盘/kaldi_dataset			#加载下载好的数据集
   data_url=www.openslr.org/resources/33
   ```

2. 修改脚本文件：修改s5下面的**cmd.sh**脚本，把原脚本注释掉，修改为本地运行：

   ```shell
   #export train_cmd=queue.pl
   #export decode_cmd="queue.pl --mem 4G"
   #export mkgraph_cmd="queue.pl --mem 8G"
   #export cuda_cmd="queue.pl --gpu 1"
   export train_cmd=run.pl
   export decode_cmd="run.pl --mem 4G"
   export mkgraph_cmd="run.pl --mem 8G"
   export cuda_cmd="run.pl --gpu 1"
   ```

3. 运行：进入kaldi/egs/aishell/s5路径下,修改权限并执行脚本**run.sh**。

   ```shell
   chnod 777 ./run.sh
   ./run.sh
   ```



### 总结：

在传统的语音识别系统中，HMM通常与Gaussian Mixture Models (GMMs) 结合使用，用来建模声学特征和音素之间的关系。具体流程如下：

1. **声学建模**：利用GMM-HMM模型估计每个音素的概率分布。
2. **语言模型**：构建一个语言模型来预测下一个词的可能性，帮助提高识别准确性。
3. **解码器**：结合声学模型和语言模型的结果，通过Viterbi算法找到最有可能的词序列。

然而，随着深度学习的发展，特别是递归神经网络（RNN）、长短时记忆网络（LSTM）以及Transformer架构的兴起，HMM-GMM逐渐被基于深度神经网络（DNN）的方法所取代，比如CTC、注意力机制（Attention Mechanism）等。这些新型方法能够直接从原始音频数据中学习特征表示，并且不需要显式的状态对齐信息。