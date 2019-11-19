{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "V8-yl-s-WKMG"
   },
   "source": [
    "# Object Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "collapsed": true,
    "id": "aSlYc3JkWKMa"
   },
   "outputs": [],
   "source": [
    "import time\n",
    "import requests\n",
    "import cv2\n",
    "import numpy as np\n",
    "import os\n",
    "import numpy as np\n",
    "import os\n",
    "import six.moves.urllib as urllib\n",
    "import sys\n",
    "import tarfile\n",
    "import tensorflow as tf\n",
    "import zipfile\n",
    "\n",
    "from distutils.version import StrictVersion\n",
    "from collections import defaultdict\n",
    "from io import StringIO\n",
    "from matplotlib import pyplot as plt\n",
    "from PIL import Image\n",
    "\n",
    "import cv2\n",
    "sys.path.append(\"..\")\n",
    "from object_detection.utils import ops as utils_ops\n",
    "%matplotlib inline\n",
    "from utils import label_map_util\n",
    "from utils import visualization_utils as vis_util\n",
    "\n",
    "# What model to download.\n",
    "MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'\n",
    "MODEL_FILE = MODEL_NAME + '.tar.gz'\n",
    "DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'\n",
    "\n",
    "# Path to frozen detection graph. This is the actual model that is used for the object detection.\n",
    "PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'\n",
    "\n",
    "# List of the strings that is used to add correct label for each box.\n",
    "PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')\n",
    "\n",
    "## Download Model\n",
    "\n",
    "# opener = urllib.request.URLopener()\n",
    "# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)\n",
    "# tar_file = tarfile.open(MODEL_FILE)\n",
    "# for file in tar_file.getmembers():\n",
    "#   file_name = os.path.basename(file.name)\n",
    "#   if 'frozen_inference_graph.pb' in file_name:\n",
    "#     tar_file.extract(file, os.getcwd())\n",
    "\n",
    "## Load a (frozen) Tensorflow model into memory.\n",
    "\n",
    "detection_graph = tf.Graph()\n",
    "with detection_graph.as_default():\n",
    "  od_graph_def = tf.GraphDef()\n",
    "  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:\n",
    "    serialized_graph = fid.read()\n",
    "    od_graph_def.ParseFromString(serialized_graph)\n",
    "    tf.import_graph_def(od_graph_def, name='')\n",
    "\n",
    "## Loading label map\n",
    "category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)\n",
    "\n",
    "## Helper code\n",
    "\n",
    "def load_image_into_numpy_array(image):\n",
    "  (im_width, im_height) = image.size\n",
    "  return np.array(image.getdata()).reshape(\n",
    "      (im_height, im_width, 3)).astype(np.uint8)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "H0_1AGhrWKMc"
   },
   "source": [
    "# Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "collapsed": true,
    "id": "jG-zn5ykWKMd"
   },
   "outputs": [],
   "source": [
    "IMAGE_SIZE = (12, 8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "collapsed": true,
    "id": "92BHxzcNWKMf"
   },
   "outputs": [],
   "source": [
    "def run_inference_for_single_image(image, graph):\n",
    "  with graph.as_default():\n",
    "    with tf.Session() as sess:\n",
    "      # Get handles to input and output tensors\n",
    "      ops = tf.get_default_graph().get_operations()\n",
    "      all_tensor_names = {output.name for op in ops for output in op.outputs}\n",
    "      tensor_dict = {}\n",
    "      for key in [\n",
    "          'num_detections', 'detection_boxes', 'detection_scores',\n",
    "          'detection_classes', 'detection_masks'\n",
    "      ]:\n",
    "        tensor_name = key + ':0'\n",
    "        if tensor_name in all_tensor_names:\n",
    "          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(\n",
    "              tensor_name)\n",
    "      if 'detection_masks' in tensor_dict:\n",
    "        # The following processing is only for single image\n",
    "        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])\n",
    "        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])\n",
    "        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.\n",
    "        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)\n",
    "        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])\n",
    "        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])\n",
    "        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(\n",
    "            detection_masks, detection_boxes, image.shape[0], image.shape[1])\n",
    "        detection_masks_reframed = tf.cast(\n",
    "            tf.greater(detection_masks_reframed, 0.5), tf.uint8)\n",
    "        # Follow the convention by adding back the batch dimension\n",
    "        tensor_dict['detection_masks'] = tf.expand_dims(\n",
    "            detection_masks_reframed, 0)\n",
    "      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')\n",
    "\n",
    "      # Run inference\n",
    "      output_dict = sess.run(tensor_dict,\n",
    "                             feed_dict={image_tensor: np.expand_dims(image, 0)})\n",
    "\n",
    "      # all outputs are float32 numpy arrays, so convert types as appropriate\n",
    "      output_dict['num_detections'] = int(output_dict['num_detections'][0])\n",
    "      output_dict['detection_classes'] = output_dict[\n",
    "          'detection_classes'][0].astype(np.uint8)\n",
    "      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]\n",
    "      output_dict['detection_scores'] = output_dict['detection_scores'][0]\n",
    "      if 'detection_masks' in output_dict:\n",
    "        output_dict['detection_masks'] = output_dict['detection_masks'][0]\n",
    "  return output_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{1: {'id': 1, 'name': 'person'},\n",
       " 2: {'id': 2, 'name': 'bicycle'},\n",
       " 3: {'id': 3, 'name': 'car'},\n",
       " 4: {'id': 4, 'name': 'motorcycle'},\n",
       " 5: {'id': 5, 'name': 'airplane'},\n",
       " 6: {'id': 6, 'name': 'bus'},\n",
       " 7: {'id': 7, 'name': 'train'},\n",
       " 8: {'id': 8, 'name': 'truck'},\n",
       " 9: {'id': 9, 'name': 'boat'},\n",
       " 10: {'id': 10, 'name': 'traffic light'},\n",
       " 11: {'id': 11, 'name': 'fire hydrant'},\n",
       " 13: {'id': 13, 'name': 'stop sign'},\n",
       " 14: {'id': 14, 'name': 'parking meter'},\n",
       " 15: {'id': 15, 'name': 'bench'},\n",
       " 16: {'id': 16, 'name': 'bird'},\n",
       " 17: {'id': 17, 'name': 'cat'},\n",
       " 18: {'id': 18, 'name': 'dog'},\n",
       " 19: {'id': 19, 'name': 'horse'},\n",
       " 20: {'id': 20, 'name': 'sheep'},\n",
       " 21: {'id': 21, 'name': 'cow'},\n",
       " 22: {'id': 22, 'name': 'elephant'},\n",
       " 23: {'id': 23, 'name': 'bear'},\n",
       " 24: {'id': 24, 'name': 'zebra'},\n",
       " 25: {'id': 25, 'name': 'giraffe'},\n",
       " 27: {'id': 27, 'name': 'backpack'},\n",
       " 28: {'id': 28, 'name': 'umbrella'},\n",
       " 31: {'id': 31, 'name': 'handbag'},\n",
       " 32: {'id': 32, 'name': 'tie'},\n",
       " 33: {'id': 33, 'name': 'suitcase'},\n",
       " 34: {'id': 34, 'name': 'frisbee'},\n",
       " 35: {'id': 35, 'name': 'skis'},\n",
       " 36: {'id': 36, 'name': 'snowboard'},\n",
       " 37: {'id': 37, 'name': 'sports ball'},\n",
       " 38: {'id': 38, 'name': 'kite'},\n",
       " 39: {'id': 39, 'name': 'baseball bat'},\n",
       " 40: {'id': 40, 'name': 'baseball glove'},\n",
       " 41: {'id': 41, 'name': 'skateboard'},\n",
       " 42: {'id': 42, 'name': 'surfboard'},\n",
       " 43: {'id': 43, 'name': 'tennis racket'},\n",
       " 44: {'id': 44, 'name': 'bottle'},\n",
       " 46: {'id': 46, 'name': 'wine glass'},\n",
       " 47: {'id': 47, 'name': 'cup'},\n",
       " 48: {'id': 48, 'name': 'fork'},\n",
       " 49: {'id': 49, 'name': 'knife'},\n",
       " 50: {'id': 50, 'name': 'spoon'},\n",
       " 51: {'id': 51, 'name': 'bowl'},\n",
       " 52: {'id': 52, 'name': 'banana'},\n",
       " 53: {'id': 53, 'name': 'apple'},\n",
       " 54: {'id': 54, 'name': 'sandwich'},\n",
       " 55: {'id': 55, 'name': 'orange'},\n",
       " 56: {'id': 56, 'name': 'broccoli'},\n",
       " 57: {'id': 57, 'name': 'carrot'},\n",
       " 58: {'id': 58, 'name': 'hot dog'},\n",
       " 59: {'id': 59, 'name': 'pizza'},\n",
       " 60: {'id': 60, 'name': 'donut'},\n",
       " 61: {'id': 61, 'name': 'cake'},\n",
       " 62: {'id': 62, 'name': 'chair'},\n",
       " 63: {'id': 63, 'name': 'couch'},\n",
       " 64: {'id': 64, 'name': 'potted plant'},\n",
       " 65: {'id': 65, 'name': 'bed'},\n",
       " 67: {'id': 67, 'name': 'dining table'},\n",
       " 70: {'id': 70, 'name': 'toilet'},\n",
       " 72: {'id': 72, 'name': 'tv'},\n",
       " 73: {'id': 73, 'name': 'laptop'},\n",
       " 74: {'id': 74, 'name': 'mouse'},\n",
       " 75: {'id': 75, 'name': 'remote'},\n",
       " 76: {'id': 76, 'name': 'keyboard'},\n",
       " 77: {'id': 77, 'name': 'cell phone'},\n",
       " 78: {'id': 78, 'name': 'microwave'},\n",
       " 79: {'id': 79, 'name': 'oven'},\n",
       " 80: {'id': 80, 'name': 'toaster'},\n",
       " 81: {'id': 81, 'name': 'sink'},\n",
       " 82: {'id': 82, 'name': 'refrigerator'},\n",
       " 84: {'id': 84, 'name': 'book'},\n",
       " 85: {'id': 85, 'name': 'clock'},\n",
       " 86: {'id': 86, 'name': 'vase'},\n",
       " 87: {'id': 87, 'name': 'scissors'},\n",
       " 88: {'id': 88, 'name': 'teddy bear'},\n",
       " 89: {'id': 89, 'name': 'hair drier'},\n",
       " 90: {'id': 90, 'name': 'toothbrush'}}"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "category_index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def image_capture(url):\n",
    "    img_resp = requests.get(url)\n",
    "    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)\n",
    "    img = cv2.imdecode(img_arr, -1)\n",
    "    cv2.imwrite('test_images\\camera_test_images\\image1.jpg',img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "## focal length of J7 prime without zoom(f) = 3.5mm\n",
    "## sensor height of J7 prime(sh) = 3.49mm\n",
    "## distance of A from Camera = f(mm)*real_height(mm)*image_height(px)\n",
    "##                             ______________________________________\n",
    "##                                  object_height(px)*sh(mm)\n",
    "\n",
    "\n",
    "def camera_distance_cm(real_height,image_height,object_height):    #real_height in mm #image_height in px #object_height in px\n",
    "  f = 3.5\n",
    "  sh = 3.49\n",
    "  distance = (f*real_height*image_height)/(object_height*sh)\n",
    "  return (distance+40)/1000                  ##returns distance in meters"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Plural Function and Objects Counter</h1>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import inflect\n",
    "p = inflect.engine()\n",
    "def plural(word,count):\n",
    "  return p.plural(word,count)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1> Simple Sentence Function </h1>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def simple_sentence(objects,info):    ####info 2D array ==> [['bottle', 'centre', 0]]\n",
    "                                      ####objects collection counter ==> Counter({'bottle': 1})\n",
    "  \n",
    "  #sentence 1\n",
    "  sentence = \"In your view, there \"\n",
    "  if len(info) == 1:\n",
    "    sentence += \"is \"\n",
    "    for i in objects:\n",
    "      sentence += str(objects[i])+\" \"+plural(i, objects[i])+\", \"\n",
    "  else:\n",
    "    sentence += \"are \"\n",
    "    j = 0\n",
    "    for i in objects:\n",
    "      if j == 0 and objects[i]==1:\n",
    "        sentence = sentence[:-4]\n",
    "        sentence +=\"is \"\n",
    "        sentence += str(objects[i])+\" \"+plural(i, objects[i])+\", \"\n",
    "        j += 1\n",
    "      else:    \n",
    "        if j==len(objects)-2:\n",
    "          sentence += str(objects[i])+\" \"+plural(i, objects[i])+\" and \"\n",
    "        else:\n",
    "          sentence += str(objects[i])+\" \"+plural(i, objects[i])+\", \"  \n",
    "        j+=1\n",
    "  sentence = sentence[:-2]\n",
    "  sentence += \". \"\n",
    "  j=0\n",
    "  \n",
    "  #sentence 2\n",
    "  for i in objects:      \n",
    "    if objects[i]==1:\n",
    "      sentence += \"The \"+ i + \" is to your \"\n",
    "      for k in range(objects[i]):\n",
    "        sentence += info[j][1]\n",
    "        j+=1\n",
    "      sentence += \". \"\n",
    "    else:\n",
    "      sentence += \"The \"+ plural(i,objects[i]) + \" are to your \"      \n",
    "      for k in range(objects[i]):\n",
    "        sentence += info[j][1]+\", \"\n",
    "        j+=1\n",
    "      sentence = sentence[:-2]\n",
    "      sentence += \". \"\n",
    "  return sentence"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Detailed Sentence Function</h1>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def detailed_sentence(objects,info):    ####info 2D array ==> [['bottle', 'centre', 0]]\n",
    "                                      ####objects collection counter ==> Counter({'bottle': 1})\n",
    "  \n",
    "  #sentence 1\n",
    "  sentence = \"In your view, there \"\n",
    "  if len(info) == 1:\n",
    "    sentence += \"is \"\n",
    "    for i in objects:\n",
    "      sentence += str(objects[i])+\" \"+plural(i, objects[i])+\", \"\n",
    "  else:\n",
    "    sentence += \"are \"\n",
    "    j = 0\n",
    "    for i in objects:\n",
    "      if j == 0 and objects[i]==1:\n",
    "        sentence = sentence[:-4]\n",
    "        sentence +=\"is \"\n",
    "        sentence += str(objects[i])+\" \"+plural(i, objects[i])+\", \"\n",
    "        j += 1\n",
    "      else:    \n",
    "        if j==len(objects)-2:\n",
    "          sentence += str(objects[i])+\" \"+plural(i, objects[i])+\" and \"\n",
    "        else:\n",
    "          sentence += str(objects[i])+\" \"+plural(i, objects[i])+\", \"  \n",
    "        j+=1\n",
    "  sentence = sentence[:-2]\n",
    "  sentence += \". \"\n",
    "  j=0\n",
    "  \n",
    "  #sentence 2\n",
    "  for i in objects:      \n",
    "    if objects[i]==1:\n",
    "      sentence += \" The \"+ i + \" is to your \"\n",
    "      direction = \"\"      \n",
    "      for k in range(objects[i]):\n",
    "        sentence += info[j][1]\n",
    "        direction += str(info[j][2])+\", \"\n",
    "        j+=1\n",
    "      sentence += \". \"\n",
    "      sentence = sentence[:-2]\n",
    "      sentence += \" at an approximate distance of \"+direction\n",
    "      sentence = sentence[:-2]+\" meters.\"\n",
    "    else:\n",
    "      sentence += \" The \"+ plural(i,objects[i]) + \" are to your \"\n",
    "      direction = \"\"\n",
    "      for k in range(objects[i]):\n",
    "        sentence += info[j][1]+\", \"\n",
    "        direction += str(info[j][2])+\", \"\n",
    "        j+=1\n",
    "      sentence = sentence[:-2]\n",
    "      sentence += \". \"\n",
    "      sentence = sentence[:-2]\n",
    "      sentence += \" at an approximate distance of \"+direction\n",
    "      sentence = sentence[:-2]+\" meters.\"    \n",
    "  return sentence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "url = \"http://192.168.43.1:8080/shot.jpg\"\n",
    "z = 1\n",
    "\n",
    "while z<4:\n",
    "  \n",
    "  start = time.time()\n",
    "  \n",
    "  image_capture(url)\n",
    "  image = Image.open(\"test_images\\camera_test_images\\image1.jpg\")\n",
    "  # the array based representation of the image will be used later in order to prepare the\n",
    "  # result image with boxes and labels on it.\n",
    "  image_np = load_image_into_numpy_array(image)\n",
    "  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]\n",
    "  image_np_expanded = np.expand_dims(image_np, axis=0)\n",
    "  # Actual detection.\n",
    "  output_dict = run_inference_for_single_image(image_np, detection_graph)\n",
    "  # Visualization of the results of a detection.\n",
    "  vis_util.visualize_boxes_and_labels_on_image_array(\n",
    "      image_np,\n",
    "      output_dict['detection_boxes'],\n",
    "      output_dict['detection_classes'],\n",
    "      output_dict['detection_scores'],\n",
    "      category_index,\n",
    "      instance_masks=output_dict.get('detection_masks'),\n",
    "      use_normalized_coordinates=True,\n",
    "      line_thickness=8)\n",
    "\n",
    "  plt.figure(figsize=IMAGE_SIZE)\n",
    "  plt.imshow(image_np)\n",
    "  \n",
    "  dim = image_np.shape\n",
    "\n",
    "  mapped = list(zip(output_dict['detection_classes'], output_dict['detection_scores'],output_dict['detection_boxes']))\n",
    "\n",
    "  real_height_array = [0,1650, 1600, 2362, 1700, 18970, \n",
    "  3200, 4366, 4520, 1371, 2133,\n",
    "  1524, 0, 914, 4876, 457, \n",
    "  121, 250, 622, 1600, 1371,\n",
    "  1219, 2743, 1524, 1371, 5500,\n",
    "  0, 1800, 914, 0, 0,\n",
    "  1450, 300, 670, 250, 1700,\n",
    "  1530, 220, 650, 1066, 317,\n",
    "  750, 2133, 711, 228, 0,\n",
    "  150, 101, 177, 180, 157,\n",
    "  254, 177, 101, 101, 80,\n",
    "  127, 150, 150, 406, 76,\n",
    "  203, 508, 914, 609, 914,\n",
    "  0, 1219, 0, 0, 50,\n",
    "  0, 400, 396, 101, 203,\n",
    "  177, 140, 254, 406, 203,\n",
    "  914, 1778, 0, 228, 304,\n",
    "  762, 177, 600, 228, 135]\n",
    "\n",
    "  global object_array\n",
    "  object_array = []\n",
    "  array = []\n",
    "  for j,k,l in mapped:\n",
    "    x = ((l[1]+l[3])/2)*dim[1]    ##768 for test image\n",
    "    y = ((l[0]+l[2])/2)*dim[0]   ##461 for test image\n",
    "\n",
    "    #l[0] --> ymin, \n",
    "    #l[1] --> xmin , \n",
    "    #l[2] --> ymax, \n",
    "    #l[3] --> ymax, \n",
    "    #x --> width, \n",
    "    #y --> height\n",
    "\n",
    "    check0 = 0\n",
    "    check1 = dim[1]/3\n",
    "    check2 = (dim[1]/3)*2\n",
    "    check3 = dim[1]\n",
    "    name = category_index[j]['name']\n",
    "    if k>= 0.5:\n",
    "      array.append(name)\n",
    "      distance = int(round(camera_distance_cm(real_height_array[j],dim[0],(l[2]-l[0])*dim[0])))\n",
    "      #print(l[0]*dim[1], l[1]*dim[1], l[2]*dim[1], l[3]*dim[1])\n",
    "      if x >= check0 and x < check1:\n",
    "        single_object = [name,\"left\",round(distance)]\n",
    "        #print(j,k,name,x,y,\"left\",(l[2]-l[0])*dim[0], distance)\n",
    "      elif x >= check1 and x < check2:\n",
    "        single_object = [name,\"centre\",distance]\n",
    "        #print(j,k,name,x,y,\"centre\",(l[2]-l[0])*dim[0], distance)\n",
    "      else:\n",
    "        single_object = [name,\"right\",distance]\n",
    "        #print(j,k,name,x,y,\"right\",(l[2]-l[0])*dim[0], distance)\n",
    "      object_array.append(single_object)\n",
    "  print(\"Next Image\")\n",
    "  info = sorted(object_array)\n",
    "  \n",
    "  from collections import Counter\n",
    "  objects = Counter(array)  \n",
    "  ########################################################\n",
    "  simple = simple_sentence(objects,info)\n",
    "  detailed = detailed_sentence(objects,info)\n",
    "  \n",
    "  if simple == \"In your view, there ar. \":\n",
    "    simple = \"No objects detected, please try a different view.\"\n",
    "    detailed = \"No objects detected, please try a different view.\"\n",
    "  else:\n",
    "    pass\n",
    "  \n",
    "  \n",
    "  print(simple)\n",
    "  print(detailed)\n",
    "  end = time.time()\n",
    "  print(\"Processing Time : \" + str(end-start)[:-13]+\" seconds\")\n",
    "  #######  Initializing Text to Speech Engine    #########\n",
    "  \n",
    "  import pyttsx3\n",
    "  engine = pyttsx3.init()\n",
    "  voices = engine.getProperty('voices')\n",
    "  engine.setProperty('voice', voices[0].id)  # voices\n",
    "  engine.say(simple)\n",
    "  engine.runAndWait()\n",
    "  \n",
    "  engine = pyttsx3.init()\n",
    "  voices = engine.getProperty('voices')\n",
    "  engine.setProperty('voice', voices[0].id)  # voices\n",
    "  engine.say(detailed)\n",
    "  engine.runAndWait()\n",
    "  #time.sleep(10)\n",
    "  z+=1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "for i in objects:\n",
    "    print (plural(i, objects[i]),objects[i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import pyttsx3\n",
    "engine = pyttsx3.init()\n",
    "voices = engine.getProperty('voices')\n",
    "engine.setProperty('voice', voices[0].id)  # changes the voice\n",
    "engine.say('In your view, there is 1 bus, 3 people and 1 truck. The bus is to your centre. The people are to your left, left, left. The truck is to your right.')\n",
    "engine.runAndWait()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "a = \"awd\"\n",
    "b = \"awd\"\n",
    "if a==b:\n",
    "  print (True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "str(time.time())[:-4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "default_view": {},
   "name": "object_detection_tutorial.ipynb?workspaceId=ronnyvotel:python_inference::citc",
   "provenance": [],
   "version": "0.3.2",
   "views": {}
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
