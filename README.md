# 钓鱼检测常见手法

1. 基于黑名单的钓鱼识别技术，基于网站黑名单存放的数据库进行识别。

当用户打开网址后，浏览器把URL 8 发送到黑名单服务器的数据库中进行查询，如果匹配到则该网页为钓鱼网页，浏览器就阻止用户访问该网页并提示，如果没有匹配到浏览器将会继续打开网页。黑名单中的存在的钓鱼网站URL，大部分为用户举报提交的。很多钓鱼网站在获得暴利后便马上关闭，更换域名和空间以躲避网监的查处， 所以钓鱼网站平均存活率都不高，所以基于黑名单的钓鱼识别技术就要求有良好的实效性。缺点是只能对URL进行简单匹配，无法检测到黑名单以外的钓鱼网站。优点是:实施简单，处理十分迅速，没有误报率。

2. 基于URL的钓鱼识别技术

部分钓鱼网站可能存在域名和官方网站相似的情况。于是浏览器便有了一个功能。基于URL的钓鱼识别技术。(该项技术误报率很大，大部分浏览器不适用该技术。)

3. 基于页面文本特征的钓鱼识别技术

由于钓鱼网站自身特有的明显特征，可以通过基于页面文本特征的钓鱼识别技术来检测钓鱼网站。首先提取页面的钓鱼特征，如文字特征和DOM（Document Object Model）结构特征，通过分析与正常官网的特征区别进行对比识别。可以通过以下几个方面提取特征：WEB页面URL，链接对象，表单，资源等等。使用该方法误报率较低，根据启发式检测技术，能识别在页面上具有高相似度的钓鱼网站。缺点为 不可避免有误报的情况出现。

4. 基于域名whois的钓鱼识别技术

任何一个域名在注册的时候都需要在域名注册商那里填写域名注册人的相关信息。那么识别钓鱼网站的时候也可以通过域名whois进行钓鱼识别。为此我收集了数个钓鱼网站进行识别，发现它们中大部分的域名whois的信息乱填的（即为连续的字母或数字或特殊含义的号码等等），还有少部分钓鱼网站域名whois信息和被钓正规网站相似。极少部分钓鱼网站隐藏了域名whois的信息。由此可得出结论：钓鱼网站也可以通过域名whois的信息进行识别。当用户访问钓鱼网站后浏览器查看域名whois信息和正常被钓网站进行对比，得出其是否为钓鱼网站的结论，返回到客户端告知用户。该识别技术误报率较低。

5. 基于网站备案识别

钓鱼网站大部分仿冒的网站为大型网站，根据国家工信部的要求，该网站肯定已经备案。且备案信息中有该企业名称。浏览器可以通过判断“网站”是否备案，备案信息是否与正常网站信息一致。看到这里，也许有人会问，备案的信息不能伪造吗？实际上，中国网站备案步骤比较复杂，公司备案需要营业执照副本、网站负责人身份证正反面、网站负责人照片、核验单等。而且是人工审核，备案时间约为20至30个工作日。钓鱼网站流动性较大，备案困难且备案信息不可伪造。使用网站备案进行识别是一个好方法。误报率几乎为零。

6. 基于图像特征的钓鱼识别技术

有些时候基于页面文本特征的钓鱼识别技术不能有效工作的时候，可以使用基于图像特征的钓鱼识别技术进行识别。有些钓鱼网站开发者对于CSS或者美工不擅长技术，于是网页大面积使用图片进行开发，关键字就在图片里面了，文本就会很少。可根据EMD算法计算图片文件与数据库中钓鱼网站截图文件之间的相似度去判断是否为钓鱼网站。基于图像特征的钓鱼识别技术，优点就是检测那些为了躲避文本特征检测识别技术的钓鱼网页。缺点是算法十分复杂，计算量大，占用空间大。

7. 基于PR(PageRank)网页级别进行判断

PR值全称为PageRank(网页级别），钓鱼网站由于流动性和临时性，网站的PR值很低，一般的钓鱼网站PR值为0.而钓鱼网站所仿的官方真实网站PR值一般很高.当发现疑似钓鱼网站的网页时，通过判断PR值来确定该网站是否为钓鱼网站也是一个好的方法。

#  Fish_learning - 应用
[web_api_4_pishi] https://github.com/HTmonster/web_api_4_pishi

```
pip3 install tldextract
pip3 install BeautifulSoup4


# predictor\predictor\tools\getFeatures.py
r = requests.get(url, timeout=(3, 7), verify=False)

# 
```

# Fish_learning - 机器学习检测

[git] https://github.com/HTmonster/Fish_learning

## 环境配置&安装

```
! 最好的办法， 安装Conda or anaconda

版本要求：

Python 3.x
Windows x64 !      # x86部分版本不支持tensorflow
linux              # 最好用Linux, 代码是在Linux上写的
tensorflow < 2.0   # 作者原版本小于2.0, 2.0后架构变了

pip install numpy
pip install pandas

# Windows install
Conda install tensorflow (如果卡顿等着就好了， 需要等很长时间)

# Linux install
pip3 install tensorflow

mkdir /opt/tensorflow/
git clone https://github.com/HTmonster/Fish_learning.git
cd Fish_learning && cp -R page_identify/data ./ && cp page_identify/train_* ./

# Test
python3 train_textCNN_w2vec.py

tensorflow.train
├── data
│   ├── NegativeFile6.csv
│   ├── negative_urls.csv
│   ├── PositiveFile6.csv
│   └── positive_urls.csv
├── page_identify
│   ├── CNN_LSTM.py
│   ├── LSTM_CNN.py
│   ├── simpleNN.py
│   ├── TextCNN.py
│   ├── train_LSTMCNN.py
│   ├── train_simpleNN.py
│   ├── train_textCNN.py
│   ├── train_textCNN_w2vec.py
│   ├── data_Processer.py
│   └── word2vec_tool.py
├── train_LSTMCNN.py
├── train_simpleNN.py
├── train_textCNN.py
└── train_textCNN_w2vec.py

tensorflow.run
├─model(以下内容为tensorflow.train项目的训练结果)
│  ├─simpleNN
│  ├─TextCNN
│  └─word2vec
└─tools


```

## 更新 & 问题
```

tf_upgrade_v2 --intree tensorflow/ --outtree tensorflow.v2/ --reportfile report.txt


# module ‘tensorflow‘ has no attribute ‘flags‘

tf.flags = tf.compat.v1.flags

# module 'tensorflow' has no attribute 'ConfigProto/Session'

1. 原版 
session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(config=session_conf)
2. 修改
session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess =tf.compat.v1.Session(config=session_conf)  

# module 'tensorflow' has no attribute 'placeholder'

！ 不要使用: import tensorflow as tf, 改用以下方法:
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# module 'tensorflow.compat.v1' has no attribute 'contrib'
原: initializer = tf.contrib.layers.xavier_initializer()
现: initializer=tf.glorot_uniform_initializer()

# ModuleNotFoundError: No module named 'gensim'
pip3 install gensim

# file
! 原有数据因为是wb无法写入，但是如果创建文件又无法解析
data = ['plato', 'stanford', 'edu', 'entries', 'aesthetics-18th-french', '', '<PADDING>', '<PADDING>', '<PADDING>', '<PADDING>']
fpath = "/opt/tensorflow/output/wordVec/trained_word2vec.model"
pickle.dump(data, open(fpath, "wb+"))

https://stackoverflow.com/questions/53158087/how-do-i-get-word2vec-to-load-a-string-problem-dict-object-has-no-attribute

https://www.google.com/search?ei=0gnrX6raKZnN-Qbzpr34Dg&q=_load_specials&oq=_load_specials&gs_lcp=CgZwc3ktYWIQA1CjIlijImCiJGgAcAB4AIABtgiIAbMPkgEFNi0xLjGYAQCgAQKgAQGqAQdnd3Mtd2l6wAEB&sclient=psy-ab&ved=0ahUKEwiqi4y1gvPtAhWZZt4KHXNTD-8Q4dUDCA0&uact=5

# AttributeError: module 'tensorflow._api.v2.train' has no attribute 'AdamOptimizer'
原：tf.train.AdamOptimizer(1e-3)
现：tf.compat.v1.train.AdamOptimizer(1e-3)

```

# tensorflow
```
# tensorflow 设置 禁用 wroing日志
方法1： export TF_CPP_MIN_LOG_LEVEL=2

方法2：
TF_CPP_MIN_LOG_LEVEL = 1 //默认设置，为显示所有信息
TF_CPP_MIN_LOG_LEVEL = 2 //只显示error和warining信息
TF_CPP_MIN_LOG_LEVEL = 3 //只显示error信息
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

方法3:
import warnings
warnings.filterwarnings("ignore")

方法4：
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
或
tf.logging.set_verbosity(tf.logging.ERROR)
或
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
```

# 参考文章
[蜜罐钓鱼] https://normshield.com/community/
[钓鱼 Feed] https://openphish.com/phishing_feeds.html
[钓鱼数据库] https://www.phishtank.com/developer_info.php
[常见钓鱼检测方案] https://www.zhihu.com/question/35424623
[Fish_learning] https://github.com/HTmonster/Fish_learning
[Fish Web Api] https://github.com/HTmonster/web_api_4_pishi