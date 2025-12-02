import math
import numpy as np
from lib.models.odtrack import build_odtrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from collections import deque


class Adaptivesearch:
    def __init__(self, init_r=1.0, alpha=0.1, beta1=0.9, beta2=0.99, epsilon=1e-8, min_lambda=0.9, max_lambda=1.2,
                 gamma=0.1):

        self.r = init_r
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 1
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.gamma = gamma
        self.prev_g = None
        self.O_prev_g = None
        self.O_m = 0
        self.O_v = 0
        self.O_t = 1
        self.O_r = init_r

    def update_search(self, S1, S2):

        if S1 > S2:
            g_t = 2 * (S1 - S2)
        else:
            g_t = 4 * (S1 - S2)

        if self.prev_g is not None and g_t * self.prev_g < 0:
            self.m *= self.gamma
            self.v *= self.gamma
            self.t = 1

        self.prev_g = g_t

        self.m = self.beta1 * self.m + (1 - self.beta1) * g_t
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g_t ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        self.r = 1.0 + self.alpha * m_hat / (v_hat ** 0.5 + self.epsilon)

        self.r = max(self.min_lambda, min(self.max_lambda, self.r))

        self.t += 1

        return self.r


class Adaptivetemplate:
    def __init__(self, init_r=1.0, alpha=0.1, beta1=0.9, beta2=0.99, epsilon=1e-8, min_lambda=0.9, max_lambda=1.2,
                 gamma=0.1):

        self.r = init_r
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 1
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.gamma = gamma
        self.prev_g = None
        self.O_prev_g = None
        self.O_m = 0
        self.O_v = 0
        self.O_t = 1
        self.O_r = init_r

    def update_template(self, S1, S2):

        if S1 > S2:
            g_t = 2 * (S1 - S2)
        else:
            g_t = 4 * (S1 - S2)

        if self.prev_g is not None and g_t * self.prev_g < 0:
            self.m *= self.gamma
            self.v *= self.gamma
            self.t = 1

        self.prev_g = g_t

        self.m = self.beta1 * self.m + (1 - self.beta1) * g_t
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g_t ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        self.r = 1.0 + self.alpha * m_hat / (v_hat ** 0.5 + self.epsilon)

        self.r = max(self.min_lambda, min(self.max_lambda, self.r))

        self.t += 1

        return self.r


class FIFOQueue:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)

    def set_max_size(self, new_size):
        if new_size <= 0:
            raise ValueError("queue must > 0")

        self.max_size = new_size
        while len(self.queue) > new_size:
            self.queue.popleft()

    def __iter__(self):
        return iter(self.queue)

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.popleft()
        else:
            raise IndexError("queue is empty")

    def peek(self):
        if len(self.queue) > 0:
            return self.queue[0]
        else:
            raise IndexError("queue is empty")

    def peek_last(self):
        if len(self.queue) > 0:
            return self.queue[-1]
        else:
            raise IndexError("queue is empty")

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.max_size

    def size(self):
        return len(self.queue)

    def __str__(self):
        return f"FIFOQueue(max_size={self.max_size}, current_size={len(self.queue)}, items={list(self.queue)})"

    def clear(self):
        self.queue = deque(maxlen=self.max_size)

class ODTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(ODTrack, self).__init__(params)
        network = build_odtrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.num_template = self.cfg.TEST.TEMPLATE_NUMBER

        self.t = 5
        self.r_search = 1.00
        self.r_template = 1.00
        # self.update = self.update_intervals
        self.update = 50
        self.sub_num = 3
        self.search_score_queue = FIFOQueue(5)
        self.initial_search_score = None
        self.initial_template_score = 1.0


        self.forgetting_factor_s = Adaptivesearch(init_r=1.00, alpha=0.05, beta1=0.9, beta2=0.99,
                                                  min_lambda=0.9,
                                                  max_lambda=1.1, gamma=0.0)
        self.forgetting_factor_t = Adaptivetemplate(init_r=1.00, alpha=0.05, beta1=0.9, beta2=0.99,
                                                    min_lambda=0.9,
                                                    max_lambda=1.1, gamma=0.0)



    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            # self.z_dict1 = template
            self.memory_frames = [template.tensors]
            self.memory_subspace_template = [template.tensors]
            self.score_list = [1.0]
            self.show_sub = [template.tensors]

        self.memory_masks = []
        self.memory_score_list = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))

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
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        box_mask_z = None
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
            # template_score_list = self.score_list.copy
            if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
                box_mask_z = torch.cat(self.memory_masks, dim=1)
        else:
            template_list, box_mask_z = self.select_memory_frames()

        if self.frame_id <= self.sub_num:
            sub_template = self.memory_subspace_template.copy()
            template_score_list = self.score_list.copy()
        else:
            sub_template, template_score_list = self.select_memory_subspace()

        with torch.no_grad():
            # out_dict = self.network.forward(template=template_list, search=[search.tensors], ce_template_mask=box_mask_z)
            out_dict = self.network.forward_gr(template=template_list, search=[search.tensors], ce_template_mask=box_mask_z,
                                                      time=self.t, r_s=self.r_search, r_t=self.r_template,
                                                      update=self.update, s_num=self.sub_num, t_num=self.sub_num, sub_template=sub_template, show_sub=self.show_sub)
        if isinstance(out_dict, list):
            out_dict = out_dict[-1]

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        pred_boxes, conf_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        self.search_score_queue.enqueue(conf_score)
        if self.initial_search_score is None and self.search_score_queue.size() == self.sub_num:
            s_list = list(self.search_score_queue)[-self.sub_num:]
            score_mean = (sum(s_list) / self.sub_num)
            template_score_mean = (sum(template_score_list) / self.sub_num)
            self.initial_search_score = score_mean
            self.initial_template_score = template_score_mean

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors
        if self.frame_id == 10 or self.frame_id == 6 or self.frame_id == 39 or self.frame_id == 20:
            self.show_sub.append(frame)
        # mask = cur_frame.mask
        if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
            frame = frame.detach().cpu()
            # mask = mask.detach().cpu()
        self.memory_frames.append(frame)
        self.memory_subspace_template.append(frame)
        self.score_list.append(conf_score)
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))
        if 'pred_iou' in out_dict.keys():      # use IoU Head
            pred_iou = out_dict['pred_iou'].squeeze(-1)
            self.memory_ious.append(pred_iou)
        #update subsapce score
        if self.frame_id % self.update == 0:
            search_score_list = list(self.search_score_queue)[-self.sub_num:]
            search_mean = (sum(search_score_list) / self.sub_num)
            self.r_search = self.forgetting_factor_s.update_search(self.initial_search_score, search_mean)
            self.initial_search_score = search_mean
            template_mean = (sum(self.score_list) / self.num_template)
            self.r_template = self.forgetting_factor_t.update_template(self.initial_template_score, template_mean)
            self.initial_template_score = template_mean

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}



    def select_memory_subspace(self):
        num_segments = self.sub_num
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_sub_template, select_score = [], []
        
        for idx in indexes:
            frames = self.memory_subspace_template[idx]
            score = self.score_list[idx]
            if not frames.is_cuda:
                frames = frames.cuda()
                score = score.cuda()
            select_sub_template.append(frames)
            select_score.append(score)
            
        return select_sub_template, select_score

    def select_memory_frames(self):
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames, select_masks, select_score = [], [], []

        for idx in indexes:
            frames = self.memory_frames[idx]
            if not frames.is_cuda:
                frames = frames.cuda()
            select_frames.append(frames)

            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = self.memory_masks[idx]
                select_masks.append(box_mask_z.cuda())

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            return select_frames, torch.cat(select_masks, dim=1)
        else:
            return select_frames, None


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

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return ODTrack
