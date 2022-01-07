import copy
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.quantization import DeQuantStub, QuantStub
from torchvision.models import alexnet


#TODO 1
# You're not supposed to write any code in this TODO, 
# but try to familiarize yourself with the code provided
# in order to answer questions in the report. 
class ImageLoader(data.Dataset):
  '''
  Class for data loading
  '''

  train_folder = 'train'
  test_folder = 'test'

  def __init__(self,
               root_dir: str,
               split: str = 'train',
               transform: torchvision.transforms.Compose = None):
    '''
    Init function for the class.

    Note: please load data only for the mentioned split.

    Args:
    - root_dir: the dir path which contains the train and test folder
    - split: 'test' or 'train' split
    - transforms: the transforms to be applied to the data
    '''
    self.root = os.path.expanduser(root_dir)
    self.transform = transform
    self.split = split

    if split == 'train':
      self.curr_folder = os.path.join(root_dir, self.train_folder)
    elif split == 'test':
      self.curr_folder = os.path.join(root_dir, self.test_folder)

    self.class_dict = self.get_classes()
    self.dataset = self.load_imagepaths_with_labels(self.class_dict)

  def load_imagepaths_with_labels(self,
                                  class_labels: Dict[str, int]
                                  ) -> List[Tuple[str, int]]:
    '''
    Fetches all image paths along with labels

    Args:
    -   class_labels: the class labels dictionary, with keys being the classes
                      in this dataset and the values being the class index.
    Returns:
    -   list[(filepath, int)]: a list of filepaths and their class indices
    '''
    img_paths = []  # a list of (filename, class index)
    for class_name, class_idx in class_labels.items():
      img_dir = os.path.join(self.curr_folder, class_name, '*.jpg')
      files = glob.glob(img_dir)
      img_paths += [(f, class_idx) for f in files]
    return img_paths

  def get_classes(self) -> Dict[str, int]:
    '''
    Get the classes (which are folder names in self.curr_folder) along with
    their associated integer index.

    Note: Assigning integer indicies 0-14 to the 15 classes.

    Returns:
    -   Dict of class names (string) to integer labels
    '''

    classes = dict()
    classes_list = [d.name for d in os.scandir(self.curr_folder) if d.is_dir()]
    classes = {classes_list[i]: i for i in range(len(classes_list))}
    return classes

  def load_img_from_path(self, path: str) -> Image:
    ''' 
    Loads the image as grayscale (using Pillow)

    Note: Not normalizing the image to [0,1]

    Args:
    -   path: the path of the image
    Returns:
    -   image: grayscale image loaded using pillow (Use 'L' flag while converting using Pillow's function)
    '''

    img = None
    with open(path, 'rb') as f:
      img = Image.open(f)
      img = img.convert('L')
    return img

  def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
    '''
    Fetches the item (image, label) at a given index

    Note: Do not forget to apply the transforms, if they exist

    Hint:
    1) get info from self.dataset
    2) use load_img_from_path
    3) apply transforms if valid

    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    '''
    img = None
    class_idx = None
    filename, class_idx = self.dataset[index]
    # load the image and apply the transforms
    img = self.load_img_from_path(filename)
    if self.transform is not None:
      img = self.transform(img)
    return img, class_idx

  def __len__(self) -> int:
    """
    Returns the number of items in the dataset

    Returns:
        int: length of the dataset
    """
    l = len(self.dataset)
    return l

#TODO 2
def get_fundamental_transforms(inp_size: Tuple[int, int],
                               pixel_mean: np.array,
                               pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the core transforms needed to feed the images to our model

  Suggestions: Resize(), ToTensor(), Normalize() from torchvision.transforms

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean of the raw dataset [Shape=(1,)]
  - pixel_std: the standard deviation of the raw dataset [Shape=(1,)]
  Returns:
  - fundamental_transforms: transforms.Compose with the fundamental transforms
  '''

  return transforms.Compose([
      ############################################################################
      # Student code begin
      ############################################################################
      torchvision.transforms.Resize(inp_size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(pixel_mean,pixel_std),
      ############################################################################
      # Student code end
      ############################################################################
  ])

#TODO 3
class SimpleNet(nn.Module):
  '''Simple Network with atleast 2 conv2d layers and two linear layers.'''

  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

    Hints:
    1. Refer to https://pytorch.org/docs/stable/nn.html for layers
    2. Remember to use non-linearities in your network. Network without
       non-linearities is not deep.
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()  # conv2d and supporting layers here
    self.fc_layers = nn.Sequential()  # linear and supporting layers here
    self.loss_criterion = None

    ############################################################################
    # Student code begin
    ############################################################################
    
    # Initializing cnn_layers and fc_layers. 
    # Please refer to the simplenet.jpg file for network structure.
    
    self.cnn_layers = nn.Sequential(nn.Conv2d(1, 10, kernel_size = 5, stride =1), nn.ReLU(), nn.MaxPool2d(3), nn.Conv2d(10, 20, kernel_size = 5, stride = 1), nn.ReLU(), nn.MaxPool2d(3))
    self.fc_layers = nn.Sequential(nn.Linear(500, 100), nn.Linear(100,15))
    
    ############################################################################
    # Student code end
    ############################################################################


    # 3.2 Setting the loss_criterion.
    #TODO 5 - Assign appropriate loss function
    ############################################################################
    # Student code begin
    ############################################################################
    
    self.loss_criterion = nn.CrossEntropyLoss(size_average = None, reduce = None, reduction = 'sum')
    
     

    ############################################################################
    # Student code end
    ############################################################################


  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Note: do not perform soft-max or convert to probabilities in this function

    Note : You will get 3D tensor for an image input from self.cnn_layers. You need 
       to process it and make it a compatible tensor input for self.fc_layers.

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ############################################################################
    # Student code begin
    ############################################################################
    tensor = self.cnn_layers(x)
    tensor = tensor.view(-1, 500)
    model_output = self.fc_layers(tensor)
    ############################################################################
    # Student code end
    ############################################################################

    return model_output

#TODO 4
def predict_labels(model_output: torch.tensor) -> torch.tensor:
  '''
  Predicts the labels from the output of the model.

  Args:
  -   model_output: the model output [Dim: (N, 15)]
  Returns:
  -   predicted_labels: the output labels [Dim: (N,)]
  '''

  predicted_labels = None
  ############################################################################
  # Student code begin
  ############################################################################
  
  predicted_labels = torch.argmax(model_output, dim = 1)
    
  ############################################################################
  # Student code end
  ############################################################################

  return predicted_labels

#NOTE: #TODO 5 is in SimpleNet class

#TODO 6
def compute_loss(model: torch.nn.Module,
                 model_output: torch.tensor,
                 target_labels: torch.tensor,
                 is_normalize: bool = True) -> torch.tensor:
  '''
  Computes the loss between the model output and the target labels

  Note: we have initialized the loss_criterion in the model with the sum
  reduction.

  Args:
  -   model: model (which inherits from nn.Module), and contains loss_criterion
  -   model_output: the raw scores output by the net [Dim: (N, 15)]
  -   target_labels: the ground truth class labels [Dim: (N, )]
  -   is_normalize: bool flag indicating that loss should be divided by the
                    batch size
  Returns:
  -   the loss value of the input model
  '''
  loss = None

  ############################################################################
  # Student code begin
  ############################################################################
  prediction = model_output.exp()/(model_output.exp().sum(-1)).unsqueeze(-1)
  loss_function_is = torch.nn.CrossEntropyLoss()
  loss = loss_function_is(model_output, target_labels)
  
  if is_normalize == 1:
    actual_loss = loss / model_output.shape[0]
    loss = actual_loss

  ############################################################################
  # Student code end
  ############################################################################

  return loss

#TODO 7
def get_optimizer(model: torch.nn.Module,
                  config: dict) -> torch.optim.Optimizer:
  '''
  Returns the optimizer for the model params, initialized according to the config.

  Note: config has a minimum of three entries. Feel free to add more entries if you want.
  But do not change the name of the three existing entries. 
  Of course, the optimizer should be operating on the model.


  Args:
  - model: the model to optimize for
  - config: a dictionary containing parameters for the config
  Returns:
  - optimizer: the optimizer
  '''

  optimizer = None

  optimizer_type = config["optimizer_type"]
  learning_rate = config["lr"]
  weight_decay = config["weight_decay"]

  ############################################################################
  # Student code begin
  ############################################################################
  if config["optimizer_type"] == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
  elif config["optimizer_type"] == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
  #optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])
  ############################################################################
  # Student code end
  ############################################################################

  return optimizer

#NOTE: #TODO 8 is tuning hyperparams for SimpleNet, included in notebook and report.

#TODO 9
def get_data_augmentation_transforms(inp_size: Tuple[int, int],
                                     pixel_mean: np.array,
                                     pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the data augmentation + core transforms needed to be applied on the train set. Put data augmentation transforms before code transforms. 

  Suggestions: Jittering(), Flipping(), Cropping(), Rotating() from torchvision.transforms.
  Need to add core transforms such as ToTensor() and Normalize() after doing augmentation.

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean of the raw dataset
  - pixel_std: the standard deviation of the raw dataset
  Returns:
  - aug_transforms: transforms.compose with all the transforms
  '''

  return transforms.Compose([
      ############################################################################
      # Student code begin
      ############################################################################
      transforms.Resize(inp_size), transforms.ColorJitter(hue=0.05, brightness = 0.2, saturation=0.6), transforms.ToTensor(), transforms.RandomHorizontalFlip(),transforms.Normalize(pixel_mean, pixel_std)
      ############################################################################
      # Student code end
      ############################################################################
  ])

#TODO 10
class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one. Otherwise the training will take a long time. To freeze a layer, set the
    weights and biases of a layer to not require gradients.

    Note: Map elements of alexnet to self.cnn_layers and self.fc_layers.

    Note: Remove the last linear layer in Alexnet and add your own layer to 
    perform 15 class classification.

    Note: Download pretrained alexnet using pytorch's API (Hint: see the import statements)

    Outline of Steps
    # Step-1 Load pre-trained alexnet model
    # Step-2 Replace the output layer of fully-connected layers.
    # Step-3 Freezing the layers by setting requires_grad=False
      # example: self.cnn_layers[idx].weight.requires_grad = False
      # You could find CNN layer indices and first two FC layer indices
      #  if you run print(my_alexnet) in the notebook.
    # Step-4 Assign loss
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ############################################################################
    # Student code begin
    ############################################################################

    # Step-1 Load pre-trained alexnet model
    
    load_model = alexnet(pretrained = True)
    self.cnn_layers = nn.Sequential(*list(load_model.children())[0][:],)
    

    # Step-2 Replace the output layer of fully-connected layers.
    
    for parameter in self.cnn_layers.parameters():
        parameter.requires_grad = False
    self.fc_layers = nn.Sequential(*list(load_model.children())[2][:-1], nn.Linear(4096, 15))

    # Step-3 Freezing the layers by setting requires_grad=False
    # example: self.cnn_layers[idx].weight.requires_grad = False
    # You could find CNN layer indices and first two FC layer indices 
    #  if you ran print(my_alexnet) in the notebook.
    
    self.fc_layers[1].weight.requires_grad = False
    self.fc_layers[1].weight.requires_grad = False
    self.fc_layers[4].weight.requires_grad = False
    self.fc_layers[4].weight.requires_grad = False
    
    # Step-4 Assign loss
    
    self.loss_criterion = nn.CrossEntropyLoss(size_average = None, reduce = None, reduction = 'sum')
    

    ############################################################################
    # Student code end
    ############################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Note: do not perform soft-max or convert to probabilities in this function

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
    ############################################################################
    # Student code begin
    ############################################################################
    x = self.cnn_layers(x)
    x = torch.flatten(x,1)
    model_output = self.fc_layers(x)

    ############################################################################
    # Student code end
    ############################################################################

    return model_output


#NOTE: #TODO 11 is tuning hyperparams for AlexNet, included in notebook and report.

###############################################################
#######  Extra Credit for 4476, Mandatory for 6476 ############
###############################################################

#TODO EC1.1
class SimpleNetDropout(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ############################################################################
    # Student code begin
    ############################################################################
    
    self.cnn_layers = nn.Sequential(nn.Conv2d(1, 10, kernel_size = 5, stride =1), nn.Dropout(0.5), nn.ReLU(), nn.MaxPool2d(3), nn.Conv2d(10, 20, kernel_size = 5, stride =1), nn.Dropout(0.25), nn.ReLU(), nn.MaxPool2d(3))
    self.fc_layers = nn.Sequential(nn. Linear(500, 100), nn. Linear(100, 15))
    self.loss_criterion = nn.CrossEntropyLoss(size_average = None, reduce = None, reduction = 'sum')     
    
    ############################################################################
    # Student code end
    ############################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Note: do not perform soft-max or convert to probabilities in this function

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ############################################################################
    # Student code begin
    ############################################################################
    x = self.cnn_layers(x)
    x = x.view(-1, 500) #flatten
    model_output = self.fc_layers(x)
    ############################################################################
    # Student code end
    ############################################################################

    return model_output

#TODO EC1.2 is tuning hyperparams for SimpleNetDropout, included in notebook and report.

#TODO EC2.1
class MyAlexNetQuantized(MyAlexNet):
  def __init__(self):
    '''
    Init function to define the layers and loss function.
    '''
    super().__init__()

    self.quant = QuantStub()
    self.dequant = DeQuantStub()

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net.

    Hints:
    1. Use the self.quant() and self.dequant() layer on input/output.

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''

    model_output = None
   
    x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
    ############################################################################
    # Student code begin
    ############################################################################
    
    # Step-1: Quantize input using quant()
    x = self.quant(x)
    
    # Step-2 : Pass the input through the model
    super().eval()
    model_output = super().forward(x)
    
    # Step-3: Dequantize output using dequant()
    model_output = self.dequant(model_output)  
    
    ############################################################################
    # Student code end
    ############################################################################

    return model_output


#TODO EC2.2
def quantize_model(float_model: MyAlexNet,
                   train_loader: ImageLoader) -> MyAlexNetQuantized:
  '''
  Quantize the input model to int8 weights.

  Args:
  -   float_model: model with fp32 weights.
  -   train_loader: training dataset.
  Returns:
  -   quantized_model: equivalent model with int8 weights.
  '''
  #quantized_model = MyAlexNet()
  #quantized_model = torch.quantization.quantize_dynamic(quantized_model, {torch.nn.Conv2d,nn.MaxPool2d,nn.Linear, nn.ReLU, nn.BatchNorm2d,nn.Flatten}, dtype=torch.qint8)


  # copy the weights from original model (still floats)
  quantized_model = MyAlexNetQuantized()
  quantized_model.cnn_layers = copy.deepcopy(float_model.cnn_layers)
  quantized_model.fc_layers = copy.deepcopy(float_model.fc_layers)

  quantized_model = quantized_model.to('cpu')

  quantized_model.eval()

  ##############################################################################
  ### Student code begin
  ##############################################################################

  # Step-1: Set up qconfig of the model
  quantized_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
  

  # Step-2: Preparing for calibration (use torch.quantization.prepare)
  torch.quantization.prepare(quantized_model, inplace = True)

  

  # Step-3: Run calibration on the training set
  # (Pass each data in training set to the prepared model)
  for image, label in train_loader:
    output = quantized_model(image)
  

  # Step-4: Do conversion (use torch.quantization.convert)
  torch.quantization.convert(quantized_model, inplace = True)

  ##############################################################################
  # Student code end
  ##############################################################################

  quantized_model.eval()

  return quantized_model
