#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import total_prediction
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


'''
如果本次页面在容易受害的名单中, 则设置为强模式 setting=1

'''
def check(url, setting=1):
    #对url进行预测
    url_predict=total_prediction.predict(url, setting)
    #print("判决结果:", url_predict)

    if url_predict ==1:
        result = {"status":"OK","result":1}
        result["zh"] = "正常网站"
    elif url_predict ==0:
        result = {"status":"OK","result":0}
        result["zh"] = "恶意网站"
    else:
        result = {"status":"ERROR","result":-1}
        result["zh"] = "未知错误"
        
    result["url"] = url
    result["predict"] = url_predict
    
    return result


if __name__=="__main__":
    #target = ""
    #f = open(target, "r")
    #for line in f:
    #    line = line.strip("\r\n\ ")
    #    check(line)
    
    target = "www.baidu.com"
    result = check(target)
    print(result)




