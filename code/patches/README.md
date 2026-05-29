`ostrack_prompt.py`和 `vit_ce_prompt`放在 `vipt/lib/models/vipt`,直接进行覆盖即可，做了很好的后向支持

`vipt_online_template.py`和 `vipt_single_template`放在 `vipt/lib/test/tracker`

在 `RGBE_workspace/test_rgbe_mgpus.py`中进行修改 `from lib.test.tracker`.`vipt_online_template import ViPTTrack`，使用原始的就用 `from vipt`，使用新增的就 `vipt_single_template`或者 `vipt_online_template`

关于如何在自己的数据集上跑，可以阅读vipt教程，进行一定的修改即可
