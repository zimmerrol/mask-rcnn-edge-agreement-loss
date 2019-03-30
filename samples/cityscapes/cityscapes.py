import glob
import os
import sys

import imgaug
from PIL import Image
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
import time
import random
import tensorflow as tf
import numpy as np
import io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from cityscapesscripts.helpers.labels import id2label
import csv


class COCOWrapper(COCO):
    def __init__(self, dataset, detection_type='bbox'):
        supported_detection_types = ['bbox', 'segmentation']
        if detection_type not in supported_detection_types:
            raise ValueError('Unsupported detection type: {}. '
                             'Supported values are: {}'.format(
                detection_type, supported_detection_types))
        self._detection_type = detection_type
        COCO.__init__(self)
        self.dataset = dataset
        self.createIndex()


# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class CityscapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific3,

    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscapes"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 2

    BACKBONE = 'resnet50'

    # Number of classes (including background)
    NUM_CLASSES = 1 + 34  # cityscapes has 80 classes

    def __init__(self, args):
        super().__init__()

        if args:
            self.RUN_NAME = args.run_name
            self.EDGE_LOSS_SMOOTHING_GT = args.edge_loss_smoothing_groundtruth
            self.EDGE_LOSS_SMOOTHING_PREDICTIONS = args.edge_loss_smoothing_predictions
            self.EDGE_LOSS_FILTERS = args.edge_loss_filters
            self.EDGE_LOSS_NORM = args.edge_loss_norm
            self.EDGE_LOSS_WEIGHT_FACTOR = args.edge_loss_weight_factor
            self.EDGE_LOSS_WEIGHT_ENTROPY = args.edge_loss_weight_entropy
            self.MASK_SHAPE = (args.mask_size, args.mask_size)


############################################################
#  Dataset
############################################################
class CityscapesDataset(utils.Dataset):
    def load_cityscapes(self, dataset_dir, subset, num_examples=None):
        """Load a subset of the cityscapes dataset.
        dataset_dir: The root directory of the cityscapes dataset.
        subset: What to load (train, val, test)
        num_examples: How many samples to load. If None, load all.
        """

        self.class_labels = {
            'ego vehicle': 1,
            'rectification border': 2,
            'out of roi': 3,
            'static': 4,
            'dynamic': 5,
            'ground': 6,
            'road': 7,
            'sidewalk': 8,
            'parking': 9,
            'rail track': 10,
            'building': 11,
            'wall': 12,
            'fence': 13,
            'guard rail': 14,
            'bridge': 15,
            'tunnel': 16,
            'pole': 17,
            'polegroup': 18,
            'traffic light': 19,
            'traffic sign': 20,
            'vegetation': 21,
            'terrain': 22,
            'sky': 23,
            'person': 24,
            'rider': 25,
            'car': 26,
            'truck': 27,
            'bus': 28,
            'caravan': 29,
            'trailer': 30,
            'train': 31,
            'motorcycle': 32,
            'bicycle': 33,
            'license plate': 34,
        }

        # Add classes
        for i in range(len(self.class_labels)):
            self.add_class("cityscapes", i, list(self.class_labels.keys())[i])

        image_paths = glob.glob(os.path.join(dataset_dir, "leftImg8bit", subset, "*", "*_leftImg8bit.png"))

        if subset == 'train':
            random.shuffle(image_paths)  # works in place
            if num_examples is not None:
                image_paths = image_paths[:num_examples]
        else:
            image_paths = sorted(image_paths)

        for imageId, imagePath in enumerate(image_paths):
            self.add_image(
                source="cityscapes",
                image_id=imageId,
                path=imagePath,
                width=2048,
                height=1024
            )

    def load_image(self, image_id):
        """Load images according to the given image ID."""
        info = self.image_info[image_id]
        image = Image.open(info['path'])
        return np.array(image)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cityscapes":
            return info
        else:
            super(CityscapesDataset, self).image_reference(image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['path']
        paths = path.split('leftImg8bit')
        anno_path = paths[0] + 'gtFine' + paths[1] + 'gtFine_instanceIds.png'
        anno = self.read_annotation_file(anno_path)

        if len(anno['type']) > 0:
            inst_masks = np.stack(anno['instance_mask'], axis=-1)  # [w, h, number_inst_masks]
            inst_classes = np.stack(anno['type'], axis=-1)
            return inst_masks, inst_classes
        else:
            # Call super class to return an empty mask
            return super(CityscapesDataset, self).load_mask(image_id)

    def read_annotation_file(self, instance_mask_path):
        """
        Reads the manual annotated scm ground truth data
        :param path_to_annotation_file:
        :return: dictionary
        """
        anno = {
            'type': [],
            'instance_mask': [],
        }
        with tf.gfile.GFile(instance_mask_path, 'rb') as fid:
            encoded_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_png)
        instance_masks = Image.open(encoded_png_io)
        instance_masks = np.asarray(instance_masks)
        instance_masks_id = np.unique(instance_masks)
        # Iterate over the instances in the frame
        for instance_id in instance_masks_id:
            # Formula to determine the pixel value given the instance id and instance type
            label = int(instance_id / 1000)
            if label > 0:
                if label in id2label:
                    labelTuple = id2label[label]
                    if labelTuple.hasInstances:
                        instance_mask_for_id = np.array(instance_masks == instance_id, dtype=np.uint8)
                        polygon = np.transpose(np.nonzero(instance_mask_for_id))
                        bbox_left = min(polygon[:, 1])
                        bbox_right = max(polygon[:, 1])
                        bbox_top = min(polygon[:, 0])
                        bbox_bottom = max(polygon[:, 0])
                        width = bbox_right - bbox_left
                        height = bbox_bottom - bbox_top

                        #if width > 20 and height > 20:
                        #    instance_area = np.sum(instance_mask_for_id)
                        #    if instance_area / (width * height) > 0.3 and instance_area > 400:  # 20 x 20 pixels
                        anno['instance_mask'].append(instance_mask_for_id)
                        anno['type'].append(label)

        return anno


def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks, gt, instance_counter):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        if rois.shape[0] is not 0:
            for i in range(rois.shape[0]):
                class_id = class_ids[i]
                score = scores[i]
                bbox = np.around(rois[i], 1)
                mask = masks[:, :, i]

                result = {
                    "image_id":
                        image_id,
                    "id":
                        i + 1 + instance_counter if gt is True else i + instance_counter,
                    "category_id":
                        dataset.get_source_class_id(class_id, "cityscapes"),
                    # Leave out to use the instance masks
                    "bbox":
                        [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "area":
                        (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]),
                    "iscrowd":
                        0,
                    "score":
                        score,
                    "segmentation":
                        maskUtils.encode(np.asfortranarray(mask))
                }
                results.append(result)

            instance_counter = instance_counter + i + 1
        else:
            print('No instances detected in groundtruth/detection')
    return results, instance_counter

def parse_eval_line(cocoEval):
    return [
        ["Precision/Recall", "IoU", "area", "maxDets", "value"],
        ["Average Precision  (AP)", "0.50:0.95", "all", "100", cocoEval.stats[0]],
        ["Average Precision  (AP)", "0.50", "all", "100", cocoEval.stats[1]],
        ["Average Precision  (AP)", "0.75", "all", "100", cocoEval.stats[2]],
        ["Average Precision  (AP)", "0.50:0.95", "small", "100", cocoEval.stats[3]],
        ["Average Precision  (AP)", "0.50:0.95", "medium", "100", cocoEval.stats[4]],
        ["Average Precision  (AP)", "0.50:0.95", "large", "100", cocoEval.stats[5]],
        ["Average Recall  (AR)", "0.50:0.95", "all", "1", cocoEval.stats[6]],
        ["Average Recall  (AR)", "0.50:0.95", "all", "10", cocoEval.stats[7]],
        ["Average Recall  (AR)", "0.50:0.95", "all", "100", cocoEval.stats[8]],
        ["Average Recall  (AR)", "0.50:0.95", "small", "100", cocoEval.stats[9]],
        ["Average Recall  (AR)", "0.50:0.95", "medium", "100", cocoEval.stats[10]],
        ["Average Recall  (AR)", "0.50:0.95", "large", "100", cocoEval.stats[11]]
    ]


def write_eval_to_csv(cocoEval, model_path):
    step_number = model_path.split('_')[-1].split('.')[0]
    csv_file_path = os.path.join(os.path.dirname(model_path),
                                 'coco_eval_type_{}_at_step_{}.csv'.format(cocoEval.params.iouType, step_number))
    with open(csv_file_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_rows = parse_eval_line(cocoEval)
        csv_writer.writerows(csv_rows)


def evaluate_coco(model, dataset, eval_type="segm", limit=1, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    '''if limit:
        image_ids = image_ids[:limit]'''

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()
    gt_instance_counter = 0
    pred_instance_counter = 0

    results = []
    groundtruth_list = []

    for i, image_id in enumerate(tqdm(image_ids)):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert groundtruth information to COCO format
        gt_masks, gt_classes = dataset.load_mask(coco_image_ids[i:i + 1][0])
        image_groundtruth, gt_instance_counter = build_coco_results(dataset, image_ids=coco_image_ids[i:i + 1],
                                                                    rois=utils.extract_bboxes(gt_masks),
                                                                    class_ids=gt_classes,
                                                                    scores=np.ones_like(gt_classes),
                                                                    masks=gt_masks, gt=True,
                                                                    instance_counter=gt_instance_counter)
        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results, pred_instance_counter = build_coco_results(dataset, coco_image_ids[i:i + 1], r["rois"],
                                                                  r["class_ids"], r["scores"],
                                                                  r["masks"].astype(np.uint8), gt=False,
                                                                  instance_counter=pred_instance_counter)

        results.extend(image_results)
        groundtruth_list.extend(image_groundtruth)

    # Check that groundtruth instance ids are always given consecutively.
    prev_instance_id = 0
    for gt in groundtruth_list:
        assert gt['id'] - prev_instance_id == 1, 'Failed'
        prev_instance_id = gt['id']

    groundtruth_dict = {
        'annotations': groundtruth_list,
        'images': [{'id': image_id, 'height': dataset.load_image(image_id).shape[0],
                    'width': dataset.load_image(image_id).shape[1]} for image_id in coco_image_ids],
        'categories': [{'id': class_id, 'name': class_name} for class_id, class_name in
                       zip(dataset.class_ids, dataset.class_names)]
    }
    coco = COCOWrapper(groundtruth_dict)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, 'segm')
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    write_eval_to_csv(cocoEval, model_path)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, 'bbox')
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    write_eval_to_csv(cocoEval, model_path)

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on cityscapes.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on cityscapes")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the cityscapes dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--edge-loss-smoothing-predictions',
                        action='store_true',
                        help='Apply a smoothing method on the predictions before calculating the edge loss.')
    parser.add_argument('--edge-loss-smoothing-groundtruth',
                        action='store_true',
                        help='Apply a smoothing method on the groundtruth before calculating the edge loss.')
    parser.add_argument('--edge-loss-filters', required=False,
                        default="",
                        nargs="*",
                        metavar='<sobel-x|sobel-y|laplace>',
                        help='List of filters to use to calculate the edge loss (default=[]]).',
                        type=str)
    parser.add_argument('--run-name', required=False,
                        default=None,
                        help='Name of the run (default=None, uses the current time).',
                        type=str)
    parser.add_argument('--edge-loss-norm', required=False, default="l2", metavar='<l1|l2|l3|l4|l5>',
                        help='Set the type of L^p norm to calculate the Edge Loss (default=l2).')
    parser.add_argument('--training-mode', default='full', metavar='<full|base|fine>', type=str,
                        help='Either perform the full training schedule or just the final finetuning of 40k steps (default=full).')
    parser.add_argument('--edge-loss-weight-entropy', action="store_true",
                        help="Use the pixel-wise edge loss to weight an additional cross entropy.")
    parser.add_argument('--edge-loss-weight-factor', default=1.0, type=float,
                        help='Scalar factor to weight the edge loss relatively to the other losses.')
    parser.add_argument('--mask-size', default=28, type=int,
                        help='Size of the masks.')

    args = parser.parse_args()

    # filter invalid edge filter values
    valid_edge_filter_values = ["sobel-x", "sobel-y", "sobel-magnitude", "laplace"]
    args.edge_loss_filters = [x for x in args.edge_loss_filters if x in valid_edge_filter_values]
    args.use_edge_loss = len(args.edge_loss_filters) > 0

    if args.training_mode not in ["full", "fine", "base"]:
        raise ValueError("training-mode must be either `full`, `base` or `fine`.")

    print("Run's name: ", args.run_name)
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Mask Size:", args.mask_size)

    # Configurations
    if args.command == "train":
        config = CityscapesConfig(args)
    else:
        class InferenceConfig(CityscapesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0


        config = InferenceConfig(args)
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # remove the current logdir first
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights & change the log dir
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        with open(os.path.join(model.log_dir, "configuration.log"), "w") as file:
            config.display(file=file)

        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CityscapesDataset()
        dataset_train.load_cityscapes(args.dataset, "train")
        dataset_train.prepare()
       
        # Validation dataset
        dataset_val = CityscapesDataset()
        dataset_val.load_cityscapes(args.dataset, "val")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        if args.training_mode == "full" or args.training_mode == "base":
            ## Training - Stage 1
            #print("Training network heads")
            #model.train(dataset_train, dataset_val,
            #            learning_rate=config.LEARNING_RATE,
            #            epochs=5,
            #            layers='heads',
            #            augmentation=augmentation)

            # Training - Stage 2
            # Finetune layers from ResNet stage 4 and up
            print("Fine tune Resnet stage 4 and up")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=18,
                        layers='all',
                        augmentation=augmentation)

            # Training - Stage 3
            # Finetune layers from ResNet stage 4 and up
            print("Fine tune Resnet stage 4 and up")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE/10,
                        epochs=24,
                        layers='all',
                        augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CityscapesDataset()
        dataset_val.load_cityscapes(args.dataset, "val")
        dataset_val.prepare()
        print("Running COCO evaluation on all validation images of cityscapes.")
        evaluate_coco(model, dataset_val, model_path, limit=None)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
