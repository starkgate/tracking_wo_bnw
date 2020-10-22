from mmdet.apis import init_detector, inference_detector


class Model():
    def __init__(self, config, checkpoint, device):
        # these values are cached to allow for feature reuse
        self.model = init_detector(config, checkpoint, device=device)
        self.preprocessed_images = None
        self.device = device

    def detect(self, img):
        # return detections, transformed img
        detections, trans_img = inference_detector(self.model, img['im_path'][0], workaround=True)
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