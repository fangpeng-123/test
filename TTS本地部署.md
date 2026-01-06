ChatTTS 本地部署与测试操作指南 ，适用于 Ubuntu 22.04，基于 Miniconda 虚拟环境的 ChatTTS 部署与测试完整操作文档。内容涵盖源码获取、环境准备、依赖安装、模型下载、推理测试。适用于大多数本地离线推理场景，模型拉取使用 Hugging Face。



### 一. 拉取源码

##### 1.1 github 拉取 ChatTTS 项目源码

- **对话式 TTS**: ChatTTS 针对对话式任务进行了优化，能够实现自然且富有表现力的合成语音。它支持多个说话者，便于生成互动式对话。适用于大型语言模型(**LLM**)助手的对话任务，以及诸如对话式音频和视频介绍等应用。

- **精细的控制**: 该模型可以预测和控制精细的韵律特征，包括笑声、停顿和插入语。

- **更好的韵律**: ChatTTS 在韵律方面超越了大多数开源 TTS 模型。我们提供预训练模型以支持进一步的研究和开发。

```bash
# 克隆原始项目
git clone https://github.com/2noise/ChatTTS 	# 这一步如果有外网的网络条件可以选择 lfs 下载，注意使用时需要提前安装

# 安装指令
sudo apt-get install git-lfs
git lfs install
```



##### 1.2 使用 Hugging Face 镜像拉取文本转语音模型

```bash
# 下载完后注意检查文件列表
git clone https://hf-mirror.com/2Noise/ChatTTS

# 文件列表对照
├── asset
│   ├── Decoder.pt
│   ├── Decoder.safetensors
│   ├── DVAE_full.pt
│   ├── DVAE.pt
│   ├── DVAE.safetensors
│   ├── Embed.safetensors
│   ├── gpt
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── GPT.pt
│   ├── spk_stat.pt
│   ├── tokenizer
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   ├── tokenizer.pt
│   ├── Vocos.pt
│   └── Vocos.safetensors
├── config
│   ├── decoder.yaml
│   ├── dvae.yaml
│   ├── gpt.yaml
│   ├── path.yaml
│   └── vocos.yaml
└── README.md
```



### 二. 配置环境

##### 2.1 conda 创建模型运行环境

服务器环境是 ubuntu 22.04 ，如需在 windows 环境下部署可选择 WSL 进行后续操作，运行环境配置方法如下：

```bash
# 安装 miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh

# 按提示操作，安装完后执行
source ~/.bashrc

# 创建虚拟环境
conda create -n chattts python=3.10 -y		# 这一步注意根据 ChatTTs 官方文档要求 python 版本不大于3.10但是网上也有使用3.11的案例
conda activate		# 切换环境
```



##### 2.2 下载依赖库

对于 torch 版本选择以及 cuda 版本的选择需要看具体使用的显卡型号。笔者测试部署使用的 CPU 进行推理。 

```bash
# 安装时注意可能在 numpy、transfoemers 版本上报错按照要求选择和是版本就行
pip install --upgrade -r requirements.txt

# 笔者选择安装 numpy-1.26.4 transformers-4.41.1，指令如下
pip install numpy==1.26.4
pip install transformers==4.41.1
```



### 三. 测试 ChatTTS

##### 3.1 编写脚本

根据官方建议，使用 源码项目下 docs/es/README.md 内的测试脚本。如果读者有兴趣可以试试运行  tests 文件夹下的文件测试效果

```python
import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["实时率等于识别花的时间除以语音本身的时间", "尝试持用不同版本的源码以及手动下载和脚本下载模型文件的方式进行模型部署"]

wavs = chat.infer(texts)

for i in range(len(wavs)):
    """
   在某些版本的torchaudio里，第一行代码能运行，但在另一些版本中，则是第二行代码能运行。
    """
    try:
        torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
    except:
        torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]), 24000)
```



##### 3.2 问题解决

- 笔者在官方网站获取的测试脚本跑起来发现 load_models() 这个函数缺失，这个函数被封装在 core.py 文件 Chat 类中，测试发现函数已经更名为 load() 且不同渠道论坛或是文章都没有说明，具体如下：

  ```bash
  # 官网的脚本
  import torch
  import ChatTTS
  from IPython.display import Audio
  
  # Initialize ChatTTS
  chat = ChatTTS.Chat()
  chat.load_models()		# 更改为 chat.load()
  
  # Define the text to be converted to speech
  texts = ["Hello, welcome to ChatTTS!",]
  
  # Generate speech
  wavs = chat.infer(texts, use_decoder=True)
  
  # Play the generated audio
  Audio(wavs[0], rate=24_000, autoplay=True)
  ```

  

- torchaudio 可能会影响项目的实际运行，更改代码解决

  ```python
  import ChatTTS
  import torch
  import torchaudio
  
  chat = ChatTTS.Chat()
  chat.load(compile=False) # Set to True for better performance
  
  texts = ["PUT YOUR 1st TEXT HERE", "PUT YOUR 2nd TEXT HERE"]
  
  wavs = chat.infer(texts)
  
  for i in range(len(wavs)):
      """
     在某些版本的torchaudio里，第一行代码能运行，但在另一些版本中，则是第二行代码能运行。
      """
      try:
          torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
      except:
          torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]), 24000)
  ```

  

- 笔者运行环境部署在 ip地址 192.168.1.28 的服务器，遇到虚拟环境中下载包但是报错。这个问题不用在意如果包导不进去尝试重新打开文件检查虚拟环境检查解释器是否正确使用。

  ```bash
  # 检查环境中下载的包
  python -m pip list
  ```

  ![截图 2025-08-28 09-59-38](/home/yls/图片/截图/截图 2025-08-28 09-59-38.png)



### 四. 调试 FastAPI 接口

##### 4.1 模型下载：

一般来说具备科学上网条件启动主程序就会自动下载带缓存模型的目录下，如果不具备网络环境则需要根据 ChatTTS 项目源码默认的设置的模型位置自主下载并放在指定文件夹内 ~/.cache/huggingface/hub/models--2Noise--ChatTTS 中再重新启动接口服务。具体模型放置目录结构如下：

```bash
# 模型下载命令
git clone https://huggingface.co/2Noise/ChatTTS

├── blobs
│   ├── 12d59e7d0af9ccfd5deb4ec01b4db3855f3d7314
│   ├── 5ea569e3431b0ed2aa1c699461017c7174d2f56d
│   ├── 8df13367906f6cd6b1f88b3cc6f1f15599b19e94
│   ├── 9c7b3d09af3f9fea19072d4a35aecee15779f51c
│   ├── b62fb7fbd3c9b91498b869b32343642d03a25fc0
│   └── be32c1231832c60ddad7e0c2e8bd027f51a183b2
├── refs
│   └── main
└── snapshots			# 用于存放模型文件
    └── 1a3c04a8b0651689bd9242fbb55b1f4b5a9aef84
        ├── asset
        │   ├── Decoder.pt
        │   ├── Decoder.safetensors
        │   ├── DVAE_full.pt
        │   ├── DVAE.pt
        │   ├── DVAE.safetensors
        │   ├── Embed.safetensors
        │   ├── gpt
        │   │   ├── config.json
        │   │   └── model.safetensors
        │   ├── GPT.pt
        │   ├── spk_stat.pt
        │   ├── tokenizer
        │   │   ├── special_tokens_map.json
        │   │   ├── tokenizer_config.json -> ../../../../blobs/b62fb7fbd3c9b91498b869b32343642d03a25fc0
        │   │   └── tokenizer.json
        │   ├── tokenizer.pt
        │   ├── Vocos.pt
        │   └── Vocos.safetensors
        └── config
            ├── decoder.yaml -> ../../../blobs/9c7b3d09af3f9fea19072d4a35aecee15779f51c
            ├── dvae.yaml -> ../../../blobs/8df13367906f6cd6b1f88b3cc6f1f15599b19e94
            ├── gpt.yaml -> ../../../blobs/be32c1231832c60ddad7e0c2e8bd027f51a183b2
            ├── path.yaml -> ../../../blobs/5ea569e3431b0ed2aa1c699461017c7174d2f56d
            └── vocos.yaml -> ../../../blobs/12d59e7d0af9ccfd5deb4ec01b4db3855f3d7314
```



##### 4.2 启动 FastAPI 主程序：

**FastAPI CLI** 是一个命令行程序，你可以用它来部署和运行你的 FastAPI 应用程序，管理你的 FastAPI 项目，等等。当你安装 FastAPI 时（例如使用 `pip install FastAPI` 命令），会包含一个名为 `fastapi-cli` 的软件包，该软件包在终端中提供 `fastapi` 命令。要在开发环境中运行你的 FastAPI 应用，你可以使用 `fastapi dev` 命令：

```bash
# 项目启动地址 examples/api/main.py
# 启动命令
fastapi dev main.py
```

![image-20250901162357775](/home/yls/.config/Typora/typora-user-images/image-20250901162357775.png)



##### 4.3 测试主程序功能：

根据 examples/api/client.py 官方的测试脚本，测试结果如下：

![image-20250901162622940](/home/yls/.config/Typora/typora-user-images/image-20250901162622940.png)
