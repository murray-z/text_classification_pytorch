# 概述
> 基于pytorch实现文本分类

# 数据格式
- label+"\t"+text
- 每行一条数据

# 运行
- 配置文件：config.py
- model_type:cnn, rnn, rcnn, rnnatt, dpcnn, han, bert
- 运行日志及结果保存在: ./logs/model_type.log
- 模型保存在： ./model/model_type.pth
```bash
python run model_type
```
