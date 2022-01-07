import numpy as np
import torch
from PIL import Image

from cv_proj3_code.cv_proj3_unit_tests.test_models import *
from cv_proj3_code.student_code import SimpleNetDropout


def test_simple_net_dropout():
  '''
  Tests the SimpleNetDropout now contains nn.Dropout
  '''
  this_simple_net = SimpleNetDropout()


  all_layers, output_dim, counter, *_ = extract_model_layers(this_simple_net)

  assert counter['Dropout'] >= 1
  assert counter['Conv2d'] >= 2
  assert output_dim == 15
