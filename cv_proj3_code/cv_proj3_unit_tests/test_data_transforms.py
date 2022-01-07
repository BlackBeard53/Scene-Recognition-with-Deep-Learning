import numpy as np
import torch
from PIL import Image

from cv_proj3_code.student_code import get_fundamental_transforms

def test_fundamental_transforms():
  '''
  Tests the transforms using output from disk
  '''

  transforms = get_fundamental_transforms(
      inp_size=(100, 50), pixel_mean=[0.5], pixel_std=[0.3])

  try:
    inp_img = Image.fromarray(np.loadtxt(
        'cv_proj3_code/cv_proj3_unit_tests/test_data/transform_inp.txt', dtype='uint8'))
    output_img = transforms(inp_img)
    expected_output = torch.load(
        'cv_proj3_code/cv_proj3_unit_tests/test_data/transform_out.pt')

  except:
    inp_img = Image.fromarray(np.loadtxt(
        '../cv_proj3_code/cv_proj3_unit_tests/test_data/transform_inp.txt', dtype='uint8'))
    output_img = transforms(inp_img)
    expected_output = torch.load(
        '../cv_proj3_code/cv_proj3_unit_tests/test_data/transform_out.pt')

  assert torch.allclose(expected_output, output_img)
