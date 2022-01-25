# 医疗问题意图识别

## 项目概述
在这个文件夹中，我利用网络上获取到的医疗问题及其意图的数据集，训练了一个简单的 textcnn 模型，来对用户的诊疗意图进行分类。

## 文件说明
- `intent_classification.ipynb` 数据处理及模型构建和训练文件。
- `model.pt` 模型训练后的权重文件
- `intent_classification_service.py` torchserve 部署文件
- `model_store` torchserve 部署文件

## 部署
使用 docker 和 torchserve 部署。

```docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(shell pwd)/model_store:/home/model_server/model_store  pytorch/torchserve:latest torchserve --start --model-store /home/model_server/model_store --models intent_classification=intent_classification.mar```


部署完成后，可以用 curl 测试一下

```curl --location --request POST 'http://127.0.0.1:8080/predictions/intent_classification' --form 'data="心脏病有什么症状？"'```

得到返回值 ```临床表现(病症表现)```。

