from mmcv.parallel import collate, scatter
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose


class Model():
    def __init__(self, config, checkpoint, device):
        # these values are cached to allow for feature reuse
        self.model = init_detector(config, checkpoint, device=device)
        self.preprocessed_images = None
        self.device = device

    # modified code from mmdetection/two_stage.py:simple_test
    def detect(self):
        proposal_list = self.model.rpn_head.simple_test_rpn(self.features, self.preprocessed_images['img_metas'][0])
        detections = self.model.roi_head.simple_test(self.features, proposal_list, self.preprocessed_images['img_metas'][0], rescale=False)[0][0]

        # split detections into bbox coordinates and respective scores
        boxes = detections[:, :4]
        scores = detections[:, 4]
        return boxes, scores

    # test the bboxes of the previous frame on the features of the current frame
    def predict_boxes(self, boxes):
        proposal_list = [boxes]
        detections = self.model.roi_head.simple_test(self.features, proposal_list, self.preprocessed_images['img_metas'][0], rescale=False)[0][0]

        assert (proposal_list[0].shape[0] == detections.shape[0]), \
            "If there aren't as many detections as there are input boxes, it may mean that mmdetection filtered out " \
            "some of the results because their score was too low. You can fix that by setting score_thr=0 in your " \
            "mmdetection config file"

        pred_boxes = detections[:, :4]
        pred_scores = detections[:, 4]
        return pred_boxes, pred_scores

    # Preprocess (augment) the image and cache its features
    def load_image(self, img):
        cfg = self.model.cfg
        device = next(self.model.parameters()).device  # model device
        # prepare data
        data = dict(img_info=dict(filename=img['img_path']), img_prefix=None)
        # build the data pipeline
        test_pipeline = Compose(cfg.data.test.pipeline)
        # we store the metadata about mmdetection's pipeline transformations in data. It will be used later
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        # scatter to specified GPU
        self.preprocessed_images = scatter(data, [device])[0]
        self.features = self.model.extract_feats(self.preprocessed_images['img'])[0]