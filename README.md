# VOT2016-evaltool
根据跟踪算法的test结果，生成评估结果图，测试集为VOT2016格式

acc_speed.py根据eval.py和test.py的eao和speed数值，生成eao-speed气泡图，这里脚本名字有歧义

heatmap.py生成heatmap热度图，需要在test.py里修改调用

trackerbenchmark.py根据test生成的txt结果文件，生成跟踪框对比图

generate_attr_eao.m根据每个算法的eao数值生成attr_eao_vot2016.mat

eao_rank_vot2016.m根据attr_eao_vot2016.mat生成eao_rank排名图

参考：

https://blog.csdn.net/laizi_laizi/article/details/120935429

https://blog.csdn.net/qq_29894613/article/details/110053078

