from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.test.utils.hann import hann2d
from lib.models.mcitrack import build_mcitrack
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box
import numpy as np
import os
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


class MCITRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MCITRACK, self).__init__(params)
        network = build_mcitrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True:  # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.debug = params.debug
        self.frame_id = 0
        # for update
        self.h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS
        if self.debug == 2:
            save_dir = "/home/kb/kb/MCITrack/vis"
            self.save_dir = os.path.join(save_dir, params.yaml_name)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPT, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPT[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPT.DEFAULT
        print("Update threshold is: ", self.update_threshold)

        if hasattr(self.cfg.TEST.UPH, DATASET_NAME):
            self.update_h_t = self.cfg.TEST.UPH[DATASET_NAME]
        else:
            self.update_h_t = self.cfg.TEST.UPH.DEFAULT
        print("Update hidden state threshold is: ", self.update_h_t)

        if hasattr(self.cfg.TEST.INTER, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.INTER[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.INTER.DEFAULT
        print("Update intervals is: ", self.update_intervals)

        if hasattr(self.cfg.TEST.MB, DATASET_NAME):
            self.memory_bank = self.cfg.TEST.MB[DATASET_NAME]
        else:
            self.memory_bank = self.cfg.TEST.MB.DEFAULT
        print("Memory_bank is: ", self.memory_bank)


        self.t = 5
        self.r_search = 1.00
        self.r_template = 1.00
        # self.update = self.update_intervals
        self.update = 70
        self.sub_num = 5
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
        if self.debug == 2:
            self.save_path = os.path.join(self.save_dir, info['seq_name'])
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                   output_sz=self.params.template_size)
        z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template
        self.score_list = [1] * self.num_template

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)] * self.num_template
        self.frame_id = 0
        self.memory_template_list = self.template_list.copy()
        self.memory_template_anno_list = self.template_anno_list.copy()
        self.memory_score_list = [1] * len(self.template_list)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        search_list = [search]

        # run the encoder
        with torch.no_grad():
            # enc_opt = self.network.forward_encoder(self.template_list, search_list, self.template_anno_list)
            enc_opt = self.network.forward_encoder_gr(self.template_list, search_list, self.template_anno_list,
                                                      time=self.t, r_s=self.r_search, r_t=self.r_search,
                                                      update=self.update, subspace_num=self.sub_num)
        # run the time neck
        with torch.no_grad():
            hidden_state = self.h_state.copy()
            encoder_out, out_neck, h = self.network.forward_neck(enc_opt, hidden_state)
        # run the decoder
        with torch.no_grad():
            out_dict = self.network.forward_decoder(feature=out_neck)

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True:  # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['size_map'],
                                                                   out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response,
                                                                   out_dict['offset_map'],
                                                                   return_score=True)
        self.search_score_queue.enqueue(conf_score)
        if self.initial_search_score is None and self.search_score_queue.is_full():
            s_list = list(self.search_score_queue)[-self.sub_num:]
            score_mean = (sum(s_list) / self.sub_num)
            self.initial_search_score = score_mean

        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # update hiden state
        self.h_state = h
        if conf_score.item() < self.update_h_t:
            self.h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS

        # update the template
        if self.num_template > 1:
            if (conf_score > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                self.memory_template_list.append(template)
                self.memory_score_list.append(conf_score)
                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor(
                                                            [self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.memory_template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                if len(self.memory_template_list) > self.memory_bank:
                    self.memory_template_list.pop(0)
                    self.memory_template_anno_list.pop(0)
                    self.memory_score_list.pop(0)
        if (self.frame_id % self.update_intervals == 0):
            assert len(self.memory_template_anno_list) == len(self.memory_template_list)
            assert len(self.memory_score_list) == len(self.memory_template_list)
            len_list = len(self.memory_template_anno_list)
            interval = len_list // self.num_template
            for i in range(1, self.num_template):
                idx = interval * i
                if idx > len_list:
                    idx = len_list
                self.template_list.append(self.memory_template_list[idx])
                self.template_list.pop(1)
                self.template_anno_list.append(self.memory_template_anno_list[idx])
                self.template_anno_list.pop(1)
                self.score_list.append(self.memory_score_list[idx])
                self.score_list.pop(1)
        if self.frame_id % self.update == 0:
            search_score_list = list(self.search_score_queue)[-self.sub_num:]
            search_mean = (sum(search_score_list) / self.sub_num)
            self.r_search = self.forgetting_factor_s.update_search(self.initial_search_score, search_mean)
            self.initial_search_score = search_mean
            template_mean = (sum(self.score_list) / self.num_template)
            self.r_template = self.forgetting_factor_t.update_template(self.initial_template_score, template_mean)
            self.initial_template_score = template_mean
        assert len(self.template_list) == self.num_template
        assert len(self.score_list) == self.num_template

        # for debug
        if self.debug == 2:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_path, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        elif self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        return {"target_bbox": self.state,
                "best_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return MCITRACK
