import torch 
import torch.nn as nn 
import torch.optim as optim 

from PIL import Image, ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2 

import os 
import re
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import matplotlib.patches as patches 

from tqdm import tqdm

# Defining a function to calculate Intersection over Union (IoU) 
def iou(box1, box2, is_pred=True): 
	if is_pred: 
		# IoU score for prediction and label 
		# box1 (prediction) and box2 (label) are both in [x, y, width, height] format 
		
		# Box coordinates of prediction 
		b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
		b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
		b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
		b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

		# Box coordinates of ground truth 
		b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
		b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
		b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
		b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

		# Get the coordinates of the intersection rectangle 
		x1 = torch.max(b1_x1, b2_x1) 
		y1 = torch.max(b1_y1, b2_y1) 
		x2 = torch.min(b1_x2, b2_x2) 
		y2 = torch.min(b1_y2, b2_y2) 
		# Make sure the intersection is at least 0 
		intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) 

		# Calculate the union area 
		box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1)) 
		box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1)) 
		union = box1_area + box2_area - intersection 

		# Calculate the IoU score 
		epsilon = 1e-6
		iou_score = intersection / (union + epsilon) 

		# Return IoU score 
		return iou_score 
	
	else: 
		# IoU score based on width and height of bounding boxes 
		
		# Calculate intersection area 
		intersection_area = torch.min(box1[..., 0], box2[..., 0])*torch.min(box1[..., 1], box2[..., 1]) 

		# Calculate union area 
		box1_area = box1[..., 0] * box1[..., 1] 
		box2_area = box2[..., 0] * box2[..., 1] 
		union_area = box1_area + box2_area - intersection_area 

		# Calculate IoU score 
		iou_score = intersection_area / union_area 

		# Return IoU score 
		return iou_score
# Non-maximum suppression function to remove overlapping bounding boxes 
def nms(bboxes, iou_threshold, threshold): 
	# Filter out bounding boxes with confidence below the threshold. 
	bboxes = [box for box in bboxes if box[1] > threshold] 

	# Sort the bounding boxes by confidence in descending order. 
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) 

	# Initialize the list of bounding boxes after non-maximum suppression. 
	bboxes_nms = [] 

	while bboxes: 
		# Get the first bounding box. 
		first_box = bboxes.pop(0) 

		# Iterate over the remaining bounding boxes. 
		for box in bboxes: 
		# If the bounding boxes do not overlap or if the first bounding box has 
		# a higher confidence, then add the second bounding box to the list of 
		# bounding boxes after non-maximum suppression. 
			if box[0] != first_box[0] or iou( 
				torch.tensor(first_box[2:]), 
				torch.tensor(box[2:]), 
			) < iou_threshold: 
				# Check if box is not in bboxes_nms 
				if box not in bboxes_nms: 
					# Add box to bboxes_nms 
					bboxes_nms.append(box) 

	# Return bounding boxes after non-maximum suppression. 
	return bboxes_nms
# Function to convert cells to bounding boxes 
def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True): 
	# Batch size used on predictions 
	batch_size = predictions.shape[0] 
	# Number of anchors 
	num_anchors = len(anchors) 
	# List of all the predictions 
	box_predictions = predictions[..., 1:5] 

	# If the input is predictions then we will pass the x and y coordinate 
	# through sigmoid function and width and height to exponent function and 
	# calculate the score and best class. 
	if is_predictions: 
		anchors = anchors.reshape(1, len(anchors), 1, 1, 2) 
		box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2]) 
		box_predictions[..., 2:] = torch.exp( 
			box_predictions[..., 2:]) * anchors 
		scores = torch.sigmoid(predictions[..., 0:1]) 
		best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1) 
	
	# Else we will just calculate scores and best class. 
	else: 
		scores = predictions[..., 0:1] 
		best_class = predictions[..., 5:6] 

	# Calculate cell indices 
	cell_indices = ( 
		torch.arange(s) 
		.repeat(predictions.shape[0], 3, s, 1) 
		.unsqueeze(-1) 
		.to(predictions.device) 
	) 

	# Calculate x, y, width and height with proper scaling 
	x = 1 / s * (box_predictions[..., 0:1] + cell_indices) 
	y = 1 / s * (box_predictions[..., 1:2] +
				cell_indices.permute(0, 1, 3, 2, 4)) 
	width_height = 1 / s * box_predictions[..., 2:4] 

	# Concatinating the values and reshaping them in 
	# (BATCH_SIZE, num_anchors * S * S, 6) shape 
	converted_bboxes = torch.cat( 
		(best_class, scores, x, y, width_height), dim=-1
	).reshape(batch_size, num_anchors * s * s, 6) 

	# Returning the reshaped and converted bounding box list 
	return converted_bboxes.tolist()
# Function to plot images with bounding boxes and class labels 
def plot_image(image, boxes): 
	# Getting the color map from matplotlib 
	colour_map = plt.get_cmap("tab20b") 
	# Getting 14 different colors from the color map for 14 different classes 
	colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 

	# Reading the image with OpenCV 
	img = np.asarray(image) 
	# Getting the height and width of the image 
	h, w, _ = img.shape 

	# Create figure and axes 
	fig, ax = plt.subplots(1) 

	# Add image to plot 
	ax.imshow(img) 

	# Plotting the bounding boxes and labels over the image 
	for box in boxes: 
		# Get the class from the box 
		class_pred = box[0]
		# Get the center x and y coordinates 
		box = box[2:] 
		# Get the upper left corner coordinates 
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2

		# Create a Rectangle patch with the bounding box 
		rect = patches.Rectangle( 
			(upper_left_x * w, upper_left_y * h), 
			box[2] * w, 
			box[3] * h, 
			linewidth=2, 
			edgecolor=colors[int(class_pred)], 
			facecolor="none", 
		) 
		
		# Add the patch to the Axes 
		ax.add_patch(rect) 
		
		# Add class name to the patch 
		plt.text( 
			upper_left_x * w, 
			upper_left_y * h, 
			s=class_labels[int(class_pred)], 
			color="white", 
			verticalalignment="top", 
			bbox={"color": colors[int(class_pred)], "pad": 0}, 
		) 
	# Display the plot 
	plt.show()

# Create a dataset class to load the images and labels from the folder 
class Dataset(torch.utils.data.Dataset): 
	def __init__( 
		self, image_dir, label_dir, anchors, 
		image_size=416, grid_sizes=[13, 26, 52], 
		num_classes=20, transform=None
	): 
		# Image and label directories 
		self.image_dir = image_dir
		self.label_dir = label_dir 
		# Image size 
		self.image_size = image_size 
		# Transformations 
		self.transform = transform 
		# Grid sizes for each scale 
		self.grid_sizes = grid_sizes 
		# Anchor boxes 
		self.anchors = torch.tensor( 
			anchors[0] + anchors[1] + anchors[2]) 
		# Number of anchor boxes 
		self.num_anchors = self.anchors.shape[0] 
		# Number of anchor boxes per scale 
		self.num_anchors_per_scale = self.num_anchors // 3
		# Number of classes 
		self.num_classes = num_classes 
		# Ignore IoU threshold 
		self.ignore_iou_thresh = 0.5

	def __len__(self): 
		""" files = [f for f in os.listdir(self.image_dir)]
		files = int(str(files[-1]).replace('IMG_foka', '').replace('.jpg', ''))
		return files """
		return len(self.label_dir)
	
	def __getitem__(self, idx):
		imageNumberOf = 10000
		while True:
			index = str(imageNumberOf+idx)
			file_name = "IMG_foka"+index.replace(index[0],"",1)
			label_path = os.path.join(self.label_dir, file_name+".txt")
			img_path = os.path.join(self.image_dir, file_name+".jpg")
			if os.path.exists(label_path):
				break
			idx += 1
		print(label_path)
		print(img_path)
		# We are applying roll to move class label to the last column 
		# 5 columns: x, y, width, height, class_label 
		bboxes = np.roll(np.loadtxt(fname=label_path, 
						delimiter=" ", ndmin=2), 4, axis=1).tolist()
		
		# Getting the image path 
		image = np.array(Image.open(img_path).convert("RGB")) 
        
		# Albumentations augmentations 
		if self.transform: 
			augs = self.transform(image=image, bboxes=bboxes) 
			image = augs["image"] 
			bboxes = augs["bboxes"] 

		# Below assumes 3 scale predictions (as paper) and same num of anchors per scale 
		# target : [probabilities, x, y, width, height, class_label] 
		targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) 
				for s in self.grid_sizes] 
		
		# Identify anchor box and cell for each bounding box 
		for box in bboxes: 
			# Calculate iou of bounding box with anchor boxes 
			iou_anchors = iou(torch.tensor(box[2:4]), 
							self.anchors, 
							is_pred=False) 
			# Selecting the best anchor box 
			anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
			x, y, width, height, class_label = box 

			# At each scale, assigning the bounding box to the 
			# best matching anchor box 
			has_anchor = [False] * 3
			for anchor_idx in anchor_indices: 
				scale_idx = anchor_idx // self.num_anchors_per_scale 
				anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
				
				# Identifying the grid size for the scale 
				s = self.grid_sizes[scale_idx] 
				
				# Identifying the cell to which the bounding box belongs 
				i, j = int(s * y), int(s * x) 
				anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 
				
				# Check if the anchor box is already assigned 
				if not anchor_taken and not has_anchor[scale_idx]: 

					# Set the probability to 1 
					targets[scale_idx][anchor_on_scale, i, j, 0] = 1

					# Calculating the center of the bounding box relative 
					# to the cell 
					x_cell, y_cell = s * x - j, s * y - i 

					# Calculating the width and height of the bounding box 
					# relative to the cell 
					width_cell, height_cell = (width * s, height * s) 

					# Idnetify the box coordinates 
					box_coordinates = torch.tensor( 
										[x_cell, y_cell, width_cell, 
										height_cell] 
									) 

					# Assigning the box coordinates to the target 
					targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 

					# Assigning the class label to the target 
					targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) #important
					print(class_label)

					# Set the anchor box as assigned for the scale 
					has_anchor[scale_idx] = True

				# If the anchor box is already assigned, check if the 
				# IoU is greater than the threshold 
				elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: 
					# Set the probability to -1 to ignore the anchor box 
					targets[scale_idx][anchor_on_scale, i, j, 0] = -1

		# Return the image and the target 
		return image, tuple(targets)


# Device 
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load and save model variable 
load_model = False
save_model = True

# model checkpoint file name 
checkpoint_file = "checkpoint.pth.tar"

# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 
ANCHORS = [ 
	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
] 
# Batch size for training 
batch_size = 32

# Learning rate for training 
leanring_rate = 1e-5

# Number of epochs for training 
epochs = 20

# Image size 
image_size = 416

# Grid cell sizes 
s = [image_size // 32, image_size // 16, image_size // 8] 

# Class labels 
class_labels = [
	"XD",
	"Foka pospolita",
    "Nerpa obraczkowana",
    "Foka szara",
    "Amfiryta lamparcia",
    "Fokowas brodaty",
    "Kapturnik morski",
    "Krabojad foczy",
    "Lodofoka grenlandzka",
    "Mirunga",
    "Mniszka",
    "Nerpa bajkalska",
    "Weddelka",
    "Foka plamista",
    "Kajgulik pregowany"
]
# Transform for training 
train_transform = A.Compose( 
	[ 
		# Rescale an image so that maximum side is equal to image_size 
		A.LongestMaxSize(max_size=image_size), 
		# Pad remaining areas with zeros 
		A.PadIfNeeded( 
			min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
		), 
		# Random color jittering 
		A.ColorJitter( 
			brightness=0.5, contrast=0.5, 
			saturation=0.5, hue=0.5, p=0.5
		), 
		# Flip the image horizontally 
		A.HorizontalFlip(p=0.5), 
		# Normalize the image 
		A.Normalize( 
			mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
		), 
		# Convert the image to PyTorch tensor 
		ToTensorV2() 
	], 
	# Augmentation for bounding boxes 
	bbox_params=A.BboxParams( 
					format="yolo", 
					min_visibility=0.4, 
					label_fields=[] 
				) 
) 

# Transform for testing 
test_transform = A.Compose( 
	[ 
		# Rescale an image so that maximum side is equal to image_size 
		A.LongestMaxSize(max_size=image_size), 
		# Pad remaining areas with zeros 
		A.PadIfNeeded( 
			min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
		), 
		# Normalize the image 
		A.Normalize( 
			mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
		), 
		# Convert the image to PyTorch tensor 
		ToTensorV2() 
	], 
	# Augmentation for bounding boxes 
	bbox_params=A.BboxParams( 
					format="yolo", 
					min_visibility=0.4, 
					label_fields=[] 
				) 
)
# Creating a dataset object 
dataset = Dataset( 
	image_dir="labeling/output/images/", 
	label_dir="labeling/output/labels/",
	grid_sizes=[13, 26, 52], 
	anchors=ANCHORS, 
	transform=test_transform 
) 

# Creating a dataloader object 
loader = torch.utils.data.DataLoader( 
	dataset=dataset, 
	batch_size=1, 
	shuffle=True, 
) 

# Defining the grid size and the scaled anchors 
GRID_SIZE = [13, 26, 52] 
scaled_anchors = torch.tensor(ANCHORS) / ( 
	1 / torch.tensor(GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
) 

# Getting a batch from the dataloader 
x, y = next(iter(loader)) 

# Getting the boxes coordinates from the labels 
# and converting them into bounding boxes without scaling 
boxes = [] 
for i in range(y[0].shape[1]): 
	anchor = scaled_anchors[i] 
	boxes += convert_cells_to_bboxes( 
			y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor 
			)[0] 

# Applying non-maximum suppression 
boxes = nms(boxes, iou_threshold=1, threshold=0.7) 

# Plotting the image with the bounding boxes 
plot_image(x[0].permute(1,2,0).to("cpu"), boxes)




