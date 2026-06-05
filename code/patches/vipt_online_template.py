import math

from lib.models.vipt import build_viptrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import vot
from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from skimage.metrics import structural_similarity as ssim

UPDATE_INTERVAL=10
SCORE_THRESHOLD=0.7

class ViPTTrack(BaseTracker):
    def __init__(self, params):
        super(ViPTTrack, self).__init__(params)
        network = build_viptrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu',weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = False #params.debug
        self.debug = params.debug
        self.frame_id = 0
        self.update_count=0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

        #动态更新配置超参数，我这里使用的宏定义进行的硬编码
        # self.update_interval = 10 #可以用None表示不更新
        # self.update_threshold = getattr(params, 'update_threshold', 0.8)  # 置信度阈值
        # ---------- 新增：模板更新参数 ----------
        self.initial_z_tensor = None #用来保存初始模板
        self.initial_mask_z=None #用来保存初始mask
        self.start_online=False
        self.online_z_tensor=None #在线更新模板
        self.online_mask_z=None #在线更新模板的mask
        self.online_score=None
        self.trust_z_list={}
        self.count=0

    def save_img(self,tensor):
        self.count+=1
        mean_6 = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view(1, 6, 1, 1)
        std_6 = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view(1, 6, 1, 1)

        img_t = tensor.to('cpu') * std_6 + mean_6
        img_t = (img_t * 255.0).clamp(0, 255).squeeze(0)  # (6, H, W)

        rgb1 = img_t[:3].permute(1, 2, 0).cpu().byte().numpy()  # RGB
        rgb2 = img_t[3:6].permute(1, 2, 0).cpu().byte().numpy()  # 可能是 IR 或另一视角

        cv2.imwrite('time/'+f"rgb_{self.count}.png", cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR))
        cv2.imwrite('time/'+f"event_{self.count}.png", cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR))
        return



    def initialize(self, image, info: dict):
        # forward the template once

        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.initial_z_tensor=template
            self.online_z_tensor = self.initial_z_tensor.clone()

            self.save_img(self.online_z_tensor)

            # self.online_score=1
            self.trust_z_list['z']=self.initial_z_tensor.clone()
            self.trust_z_list['score']=SCORE_THRESHOLD

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.initial_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)
            self.online_mask_z=self.initial_mask_z.clone()


        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        self.update_count+=1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)


        with torch.no_grad():
            if self.start_online:
                out_dict = self.network.forward(
                    template=self.initial_z_tensor, search=search, ce_template_mask=self.initial_mask_z,
                    online_template=self.online_z_tensor,online_ce_mask=self.online_mask_z)
            else:
                out_dict = self.network.forward(
                    template=self.initial_z_tensor, search=search, ce_template_mask=self.initial_mask_z,
                    online_template=None,online_ce_mask=None)


        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        if max_score>0.7:
            self.start_online=True

        # ---------- 模板动态更新----------
        if self.update_count > UPDATE_INTERVAL and max_score > SCORE_THRESHOLD:

            # 从当前帧裁剪目标区域+生成先验mask
            target_patch_arr, target_resize_factor, target_amask_arr = sample_target(image, self.state,
                                                   self.params.template_factor,
                                                   output_sz=self.params.template_size)
            target_tensor = self.preprocessor.process(target_patch_arr)

            #计算一个score，我们的在线模板和最信任模板的相似性
            im1 = self.trust_z_list['z'].squeeze(0).cpu().numpy()[:3] # [C, H, W]
            im2 = target_tensor.squeeze(0).cpu().numpy()[:3]  # [C, H, W]
            # 将通道维移到最后：[H, W, C]
            im1 = im1.transpose(1, 2, 0)
            im2 = im2.transpose(1, 2, 0)

            # 调用 SSIM（win_size 自动取 min(H,W)，一般 >=7）
            if ssim(im1, im2, channel_axis=-1,data_range=1.0)>SCORE_THRESHOLD:
            # if True:
                self.save_img(self.online_z_tensor)
                self.update_count = 0
                self.online_z_tensor=target_tensor

                if max_score>self.trust_z_list['score']:
                    self.trust_z_list['z']=target_tensor
                    # self.trust_z_list['score']=max_score

                # self.online_score=score
                if self.cfg.MODEL.BACKBONE.CE_LOC:
                    new_bbox = self.transform_bbox_to_crop(self.state, target_resize_factor,
                                                           target_tensor.device).squeeze(1)
                    self.online_mask_z = generate_mask_cond(self.cfg, 1, target_tensor.device, new_bbox)


        # for debug，结果可视化
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return ViPTTrack
