import warnings

import torch
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose


class Model():
    def __init__(self, config, checkpoint, device):
        # these values are cached to allow for feature reuse
        self.model = init_detector(config, checkpoint, device=device)
        self.preprocessed_images = None
        self.device = device

    def custom_mmdetection_inference_detector(self, model, img):
        """Modified version of inference_detector in mmdet.apis.inference: mmdetection doesn't return the preprocessed image through the test pipeline
        This function returns the prediction AND the preprocessed image data
        Original description: Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
                images.

        Returns:
            If imgs is a str, a generator will be returned, otherwise return the
            detection results directly.
        """
        cfg = model.cfg
        device = next(model.parameters()).device  # model device
        # prepare data
        data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        test_pipeline = Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            # Use torchvision ops for CPU mode instead
            for m in model.modules():
                if isinstance(m, (RoIPool, RoIAlign)):
                    if not m.aligned:
                        # aligned=False is not implemented on CPU
                        # set use_torchvision on-the-fly
                        m.use_torchvision = True
            warnings.warn('We set use_torchvision=True in CPU mode.')
            # just get the actual data from DataContainer
            data['img_metas'] = data['img_metas'][0].data

        # forward the model
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]

        return result, data

    def detect(self, img):
        # return detections, transformed img
        detections, trans_img = self.custom_mmdetection_inference_detector(self.model, img['im_path'][0])
        self.preprocessed_images = trans_img

        return detections, trans_img

    # TODO use boxes used as input argument instead of detecting new ones?
    # What is the point since the features will need to be recalculated?
    def predict_boxes(self, boxes):
        # calling roi_head uses forward_train by default, which causes a CudaIllegalMemoryAccess error
        # we don't want to train anyways, so we'll use aug_test: augment then test
        # ndarray: x1 y1 x2 y2 score
        predictions = self.model.aug_test(self.preprocessed_images['img'], self.preprocessed_images['img_metas'])
        pred_boxes = predictions[0][0][:, :4]
        pred_scores = predictions[0][0][:, 4]

        return pred_boxes, pred_scores