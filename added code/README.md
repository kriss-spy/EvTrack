ostrack_prompt.py和vit_ce_prompt放在ViPT-main\lib\models\vipt,直接进行覆盖即可，做了很好的后向支持

vipt_online_template.py和vipt_single_template放在ViPT-main\lib\test\tracker 

在RGBE_workspace/test_rgbe_mgpus.py中进行修改from lib.test.tracker.vipt_online_template import ViPTTrack，使用原始的就用from vipt，使用新增的就vipt_single_template或者vipt_online_template

关于如何在自己的数据集上跑，可以阅读vipt教程，进行一定的修改即可