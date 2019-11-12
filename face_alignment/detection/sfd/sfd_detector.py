import os
import cv2
from torch.utils.model_zoo import load_url

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *
import numpy as np

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        # Initialise the face detector
        if path_to_detector is None:
            model_weights = load_url(models_urls['s3fd'])
        else:
            model_weights = torch.load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector = torch.nn.DataParallel(self.face_detector) # TODO 多 gpu
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)
        bboxlist = detect(self.face_detector, image, device=self.device)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]
        return bboxlist
    
    def detect_from_image_list(self, tensor_or_path_list): # TODO
        image_batch = np.stack([self.tensor_or_path_to_ndarray(e) for e in tensor_or_path_list])
        bboxlists = detect_batch(self.face_detector, image_batch, device=self.device)
        def decode_box(bboxlist):
            keep = nms(bboxlist, 0.3)
            bboxlist = bboxlist[keep, :]
            bboxlist = [x for x in bboxlist if x[-1] > 0.5]
            return bboxlist

        result = [decode_box(e) for e in bboxlists]
        return result

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
