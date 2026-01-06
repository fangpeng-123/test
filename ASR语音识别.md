## 概述：

- **特征提取**：使用librosa将音频转为帧特征（如MFCC）。
- **声学模型推理**：每帧输出对所有标签的概率分布。
- **语言模型打分**：对候选词序列加语言概率。

- **解码**：用greedy或beam search结合声学得分和语言模型得分，选最优文本输出。

### 特征提取：

- 预处理对输入语音进行降噪、静音段检测、预加重滤波等操作，提升语音信号质量。这一步可以减少环境噪声影响，并将音频切分成适合处理的帧，为特征提取做准备。

- 读取一个wav格式的文件，波形在时域上几乎没有描述能力，因此必须将波形作变换处理，常见的一种变换方法是提取MFCC特征，根据人耳的生理特性，把每一帧波形变成一个多维向量，可以简单地理解为这个向量包含了这帧语音的内容信息。这个过程叫做声学特征提取。
  - MFCC提取流程包括：语音预加重、分帧加窗，计算每帧的功率谱，经过梅尔滤波器银行求和取对数，再经离散余弦变换（DCT）得到倒谱系数。这些系数模拟了人耳对不同频率的感知特点（梅尔刻度），在压缩数据维度的同时保留了区分音素所需的关键频谱信息。

- 音频帧处理和特征提取：


```python
def extract_features(audio_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # 转置: (帧数, 特征维度)
    return mfcc.T
```



### 声学模型：

声学模型推理得到一系列概率分布或得分矩阵

- 音频预处理：音频被分帧提取特征（如梅尔频率倒谱系数 MFCC）。
- 向前传播：每一帧特征输入声学模型（如CNN、RNN、Transformer等）。
- 输出概率分布（得分矩阵）：对于每一帧，模型输出所有声学单元的概率或分数。例如，假设有三个音素[a, b, c]

- 声学模型输出每一帧（或每一段）音频上所有声原单元（音素，音节，字）上的概率分布
  - [a: 0.7, b: 0.2, c: 0.1]
  - [a: 0.2, b: 0.6, c: 0.2]

```python
class SimpleAcousticModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=-1)  # 输出每帧的log概率
```



### 语言模型：

语言模型是一种用于预测文本序列中下一个词或字符的概率分布的模型。它可以捕获语言结构的某些方面，如语法、句式和上下文信息。传统的语言模型通常使用N-gram方法、隐藏马尔可夫模型（HMM）或神经网络语言模型（如RNN、Transformer），但这些模型往往不能捕捉到长距离依赖和复杂的语义信息。

- 语言模型的核心任务是计算词序列的概率P(w1,w2,...,wn)，即判断某个词序列是否符合目标语言的语法和语义规则。例如：
  - 输入声学模型的候选发音为 ["k-a-t", "d-o-g"]，语言模型会判断 "cat" 和 "dog" 的组合是否合理（如 "cat dog" 可能比 "dog cat" 更常见）。
  - 对于同音异义词（如中文“权利” vs. “权力”，英文“see” vs. “sea”），语言模型根据上下文选择更合理的词。

- 常用方法n-gram语言模型：

  N-gram语言模型是一种基础的语言模型，用于预测下一个词或字符出现的概率，基于前N-1个词或字符。该模型将文本看作是词或字符的有序序列，并假设第n个词仅与前N-1个词相关。比如，在一个bigram（2-gram）模型中，每个词的出现只依赖于它前面的一个词。例如，"我吃"之后是"苹果"的概率可以表示为P(苹果|我吃)。

  - 优点：只需要统计词频和条件词频，不需要复杂的算法
  - 缺点：稀疏型问题随着N的增加，模型需要的存储空间急剧增加，而大多数N-gram组合在实际数据中可能并不存在，读者有兴趣可以学习一下平滑法用于解决n-gram语言模型稀疏型问题。上下文限制只能捕捉到N-1个词的上下文信息。

```python
# 假设词表
id2token = {0: 'ni', 1: 'hao', 2: 'ma', 3: 'nihao'}
token2id = {v: k for k, v in id2token.items()}

# 简单bigram语言模型分数
lm_scores = {
    ('ni', 'hao'): -0.5,
    ('hao', 'ma'): -0.3,
    ('nihao', 'ma'): -0.2,
}

def language_model_score(token_seq):
    score = 0.0
    for i in range(len(token_seq)-1):
        bigram = (token_seq[i], token_seq[i+1])
        score += lm_scores.get(bigram, -1.0)  # 未登录bigram给较低分
    return score
```



### 解码：

- 解码器综合声学模型概率和语言模型概率，在所有可能的文字序列构成的搜索空间中找到最可能的识别结果。通常使用维特比算法或束搜索算法高效地完成这一步骤，并输出最终的转写文本。解码过程中还会用到发音词典将声学模型的输出单元（如音素）映射为具体词汇。

- 使用词典、有限状态转移器（FST）图的解码过程举例

  |  ID  | 拼音/词 | 中文 |         备注         |
  | :--: | :-----: | :--: | :------------------: |
  |  0   |   ni    |  你  |         单字         |
  |  1   |   hao   |  好  |         单字         |
  |  2   |   ma    |  吗  |         单字         |
  |  3   |  nihao  | 你好 | 组合词 = "你" + "好" |
  |  4   |   hai   |  还  |         单字         |
  |  5   |  haoma  | 好吗 | 组合词 = "好" + "吗" |
  |  5   |   hao   |  号  |  多音字，同音不同意  |

  - HL.fst结构简化状态图

    (S) --0/ni--> (A) --1/hao--> (B) --2/ma--> (T)         	// 你 好 吗
     |              |                |
     |              |                |
     |              +--6/hao--> (E)  |                   			// 号
     |                               +--5/haoma--> (F)  		      // 号码
     |              
     +--3/nihao--> (C) --2/ma--> (T)                     		// 你好 吗
     |
     +--4/hai--> (D) --1/hao--> (G)                              	  // 还 好

- 语言模型（LM）处理声学模型发送的序列，可以理解为声学模型发送序列与有限状态转换器的加权和。
  $$
  totalscore =  α × 声学得分 + β × 语言模型得分
  $$
  注：其中α、β是权重参数（调节两部分的影响力）

  - ##### WFST解码：解码时联合搜索

    - 输入：声学模型输出的概率矩阵

    - 解码器：WFST解码器（HLG, HL.fst等），会把声学概率，词典，语言模型(n-gram LM)组合成一个加权有向图。

    - 求总代价最小的词序列（最优路径搜索），对应求加权和

  - ##### 流式/端到端(如Transformer/CTC/Attention 解码)

    - 解码器：使用n-gram或神经LM，如RNN-LM，Transformer-LM在beam search过程中为每个侯选序列赋值、排序、剪枝
    - 处理：解码器会生成候选输出序列，每步扩展时都考虑声学模型和语言模型分数，输出高分序列

  - ##### 假设我们有两个候选文本输出：

    - 候选1：ni hao ma

    - 候选2：ni hao la

    - 声学模型和语言模型的打分（假设都是对数概率，越大越好）如下：

      |   候选    | 声学得分（logP_声学） | 语言模型得分（logP_语言） |
      | :-------: | :-------------------: | :-----------------------: |
      | ni hao ma |         -2.0          |           -1.0            |
      | nihao ma  |         -1.0          |           -2.2            |

      假设我们设定权重：α=1，β=1

    - 计算得分：

      ni hao ma：总分 = 1 × (-2.0) + 1 × (-1.0) = -3.0

      ni hao la：总分 = 1 × (-1.5) + 1 × (-2.2) = -3.7

    所以，最终会选得分更高（数值更大，-3.0 > -3.7）的“ni hao ma”。

```python
def greedy_decode(acoustic_log_probs):
    # 每帧选概率最大的标签
    pred_ids = acoustic_log_probs.argmax(axis=1)
    return [id2token[i] for i in pred_ids]

def combined_score(acoustic_log_probs, token_seq, alpha=1.0, beta=1.0):
    # acoustic_log_probs: (帧数, 类别数)
    acoustic_score = 0.0
    for i, token in enumerate(token_seq):
        tid = token2id[token]
        acoustic_score += acoustic_log_probs[i, tid]
    lm_score = language_model_score(token_seq)
    return alpha * acoustic_score + beta * lm_score

# 简单的beam search解码
def beam_search_decode(acoustic_log_probs, beam_width=3, alpha=1.0, beta=1.0):
    T, V = acoustic_log_probs.shape
    beams = [([], 0.0)]  # (token_seq, total_score)
    for t in range(T):
        new_beams = []
        for seq, score in beams:
            for vid in range(V):
                new_seq = seq + [id2token[vid]]
                # 只在最后一步加语言模型
                cur_score = score + alpha * acoustic_log_probs[t, vid]
                if t == T-1:
                    cur_score += beta * language_model_score(new_seq)
                new_beams.append((new_seq, cur_score))
        # 保留得分最高的beam_width条路径
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    # 返回最佳路径
    return beams[0][0]
```



