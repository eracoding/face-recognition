# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import glob
import onnxruntime
from .arcface_onnx import *
from .retinaface import *

__all__ = ['get_model']


class PickableInferenceSession(onnxruntime.InferenceSession):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        model_path = values['model_path']
        self.__init__(model_path)

class ModelRouter:
    def __init__(self, onnx_file):
        self.onnx_file = onnx_file

    def get_model(self, **kwargs):
        options = onnxruntime.SessionOptions()

        options.enable_mem_pattern = False

        session = PickableInferenceSession(self.onnx_file, sess_options=options, **kwargs)

        logging.info(f'Applied providers: {session._providers}, with options: {session._provider_options}')

        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        outputs = session.get_outputs()

        logging.info(f'onnx_file={self.onnx_file}, len(outputs)={len(outputs)}')

        if len(outputs)>=5:
            return RetinaFace(model_file=self.onnx_file, session=session)
        elif input_shape[2]==112 and input_shape[3]==112:
            return ArcFaceONNX(model_file=self.onnx_file, session=session)
        else:
            return None

def find_onnx_file(dir_path):
    if not os.path.exists(dir_path):
        return None
    paths = glob.glob("%s/*.onnx" % dir_path)
    if len(paths) == 0:
        return None
    paths = sorted(paths)
    return paths[-1]

def get_model(name, **kwargs):
    root = kwargs.get('root', '~/.insightface')
    root = os.path.expanduser(root)
    model_root = osp.join(root, 'models')
    if not name.endswith('.onnx'):
        model_dir = os.path.join(model_root, name)
        model_file = find_onnx_file(model_dir)
        if model_file is None:
            return None
    else:
        model_file = name

    assert osp.exists(model_file), 'model_file should exist'
    assert osp.isfile(model_file), 'model_file should be file'
    router = ModelRouter(model_file)
    model = router.get_model(providers=['CUDAExecutionProvider'], provider_options=kwargs.get('provider_options'))
    return model
