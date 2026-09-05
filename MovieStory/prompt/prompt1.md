请你基于/home/liuzhirui/Project/MovieStory/design/pipeline_design.md中的设计原则，并详细分析/home/liuzhirui/model/Qwen3-VL-main和/home/liuzhirui/model/Wan2.2中的相关基础代码，完成下面的任务：
1. 结合/home/liuzhirui/Project/MovieStory/design/pipeline_design.md中的设计原则，将其中的第一个“3-router planner”层的设计单独提出来写成一个设计的markdown文档，因为我希望是通过增量的方式来验证模型模块的有效性。

2. 编写关于3-router-planner的设计文档后再基于设计文档编写一个代码相
  关的实现文档，别忘了结合/home/liuzhirui/model/Qwen3-VL-
  main和/home/liuzhirui/model/Wan2.2中的相关基础代码进行设
  计(训练相关基础代码参考/home/liuzhirui/model/Wan2.2/scripts-metaquery-single/train/train_metaquery_wan_new.py和/home/liuzhirui/model/Wan2.2/scripts-metaquery-single/train/train_metaquery_wan.py和/home/liuzhirui/model/Wan2.2/scripts-metaquery-single/train/train_connector_for_wan.py 和 /home/liuzhirui/model/Wan2.2/scripts-metaquery-single/train/train_stage1_openvid_local_metaquery_overfit20_ti2v_frame.sh)

3. 如果基于设计文档最终编写原模型的基础上（metaquery结合qwen3vl结合wan）增加了相关代码(3-router planner的相关代码)的话，放到/home/liuzhirui/Project/MovieStory/code的文件夹下。

4. 我希望增加这个模块后的模型能够类似/home/liuzhirui/model/Wan2.2/scripts-metaquery-single/train/train_stage1_openvid_local_metaquery_overfit20_ti2v_frame.sh 基于数据集/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M中的前100条数据进行训练，代码里面设置保存对应的checkpoint放到/home/liuzhirui/Project/MovieStory/code/checkpoint下。