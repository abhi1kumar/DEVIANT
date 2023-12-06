import os
import tqdm

import torch
import numpy as np
import subprocess

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.rpn_util import *
class Tester(object):
    def __init__(self, cfg, model, data_loader, logger, results_test= 'result_test'):
        self.cfg = cfg
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Output directory is of name result_kitti / result_nusc_kitti / result_waymo
        self.eval_dataset = cfg['dataset']['eval_dataset'] if 'eval_dataset' in cfg['dataset'].keys() else cfg['dataset']['type']
        results_test      = 'result_' + self.eval_dataset
        self.output_dir = os.path.join(cfg['trainer']['log_dir'], results_test)
        self.get_backbone_features = False if 'get_backbone_features'  not in self.cfg['tester'].keys() else self.cfg['tester']['get_backbone_features']
        self.saving_key = 'val_scale'
        self.cfg_test = cfg['tester']
        self.tester_metrics = 'kitti' if 'tester_metrics' not in self.cfg_test.keys() else self.cfg_test['tester_metrics']

        if self.cfg['tester'].get('resume_model', None):
            load_checkpoint(model = self.model,
                        optimizer = None,
                        filename = cfg['tester']['resume_model'],
                        logger = self.logger,
                        map_location=self.device)

        self.model.to(self.device)


    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        # from fvcore.nn import FlopCountAnalysis, flop_count_table
        # input = torch.zeros((1, 3, 384, 1280)).float().cuda()
        # flops = FlopCountAnalysis(self.model.backbone.cuda(), input)
        # print(flop_count_table(flops))

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.data_loader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.data_loader):
            # load evaluation data and move data to current device.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)

            # the outputs of centernet
            outputs = self.model(inputs,coord_ranges,calibs,K=50,mode='test')

            if self.get_backbone_features:
                for level in range(1):#(len(outputs)):
                    outputs_level = outputs[level].cpu().clone().detach().numpy().astype(np.float16)
                    if batch_idx == 0:
                        results[level] = outputs_level
                    else:
                        results[level] = combine(results[level], outputs_level)
                progress_bar.update()
                if batch_idx > 50:
                    break
                continue

            dets = extract_dets_from_outputs(outputs=outputs, K=50)
            dets = dets.detach().cpu().numpy()


            # get corresponding calibs & transform tensor to numpy
            calibs = [self.data_loader.dataset.get_calib(index)  for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.data_loader.dataset.cls_mean_size
            dets = decode_detections(dets = dets,
                                     info = info,
                                     calibs = calibs,
                                     cls_mean_size=cls_mean_size,
                                     threshold = self.cfg['tester']['threshold'])
            results.update(dets)
            progress_bar.update()

        if self.get_backbone_features:
            npy_folder_name = os.path.join(self.output_dir, self.saving_key)
            os.makedirs(npy_folder_name, exist_ok= True)
            print("Saving to {}".format(npy_folder_name))
            for level in range(len(results)):
                print(results[level].shape)
                save_numpy(os.path.join(npy_folder_name, 'level_' + str(level) + '.npy'), results[level])
            return


        # save the result for evaluation.
        self.save_results(results, self.output_dir)
        progress_bar.close()

        results_folder = os.path.join(self.output_dir, 'data')

        # get MAE stats
        if self.eval_dataset == "kitti":
            gt_folder = "data/KITTI/training/label_2"
        elif self.eval_dataset == "nusc_kitti":
            gt_folder = "data/nusc_kitti/validation/label"
        elif self.eval_dataset == "waymo":
            gt_folder = "data/waymo/validation/label"
        else:
            raise NotImplementedError
        get_MAE(results_folder = results_folder, gt_folder= gt_folder, conf= None, use_logging= True, logger= self.logger)

        # Now run evaluation code
        if self.tester_metrics == 'kitti':
            evaluate_kitti_results_verbose(data_folder= "data/KITTI/training/label_2", test_dataset_name= "val1",\
                                       results_folder= results_folder, conf= self.cfg,\
                                       use_logging= True, logger= self.logger)
        elif self.tester_metrics == 'waymo':
            evaluate_waymo_results_verbose(results_folder= results_folder, conf= self.cfg,\
                                   use_logging= True, logger= self.logger)
        elif self.tester_metrics == 'nusc_kitti':
            evaluate_nusc_kitti_results_verbose(results_folder= results_folder, conf= self.cfg,\
                                   use_logging= True, logger= self.logger)

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)
        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()







