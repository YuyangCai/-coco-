在训练好自己的数据集之后，我们会希望获取类似COCO形式的评价指标

get_cocometrics1.py将自定义的数据集生成所需的标注json文件

get_cocometrics2.py用于将验证后的模型json文件保存

get_cocometrics3.py将对比1和2的json文件，进行计算COCO形式评价指标