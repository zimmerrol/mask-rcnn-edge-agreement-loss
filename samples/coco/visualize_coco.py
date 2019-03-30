import argparse
import os
import sys

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops

import samples.coco.coco as coco
from mrcnn import model as modellib
from mrcnn import visualize, utils
from mrcnn.model import log

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

MODEL_DIR = os.path.join(ROOT_DIR, "logs/eval_logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
ROOT_DIR = os.path.abspath("../../")
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0 -> Force eval on cpu, to not intervene with gpu trainings

parser = argparse.ArgumentParser()
parser.add_argument("--coco_dir", type=str, help="root directory of coco 2014 data set.", required=True)
parser.add_argument("--model_checkpoint", type=str, help="path to checkpoint of model1", required=True)
parser.add_argument("--action", choices=["feature_maps", "visualize_masks", "compare_instance_masks"],
                    help="feature_maps: Visualize a selection of feature_maps of model1\n"
                         "visualize_masks: Visualize instance masks by running inference on images from coco\n"
                         "compare_instance_masks: Highlight the difference of predicted and groundtruth instance mask.",
                    required=True)
parser.add_argument("-l", "--image_ids", nargs='+', help="List of COCO images to visualize.",
                    default=[3743])
parser.add_argument("--output_dir", type=str, help="Output directory of visualizations", required=True)
args = parser.parse_args()


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def trim(im, bg="black"):
    if bg == "white":
        bg = Image.new(im.mode, im.size, im.getpixel((255, 255)))
    else:
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def trim2(img):
    def remove_color_alongaxis(img, color, axis):
        sum_axis = np.sum(img, axis=axis)
        first_coor = 0

        width, height, _ = img.shape
        for row in range(width):
            sum_of_pixels = np.max(sum_axis[row])
            if sum_of_pixels == 255 * 1600:
                first_coor = row
            else:
                break

        for row in range(width):
            sum_of_pixels = np.max(sum_axis[row])
            if sum_of_pixels == 255 * 1600:
                first_coor = row
            else:
                break

    # Remove white
    img_np = np.array(img)

    # Remove black


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    plt.tight_layout()
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def prepare_coco(coco_dir):
    # Override the training configurations with a few
    # changes for inferencing.
    config = coco.CocoConfig(args=None)

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig(args=None)
    config.display()

    dataset = coco.CocoDataset()
    dataset.load_coco(coco_dir, "minival")
    dataset.prepare()

    return config, dataset


def load_model(path, config):
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    # Load weights
    print("Loading weights ", path)
    model.load_weights(path, by_name=True)
    return model


def visualize_feature_map(model, dataset, config, image_ids, output_dir):
    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        # Get activations of a few sample layers
        activations = model.run_graph([image], [
            ("input_image", model.keras_model.get_layer("input_image").output),
            ("res4w_out", model.keras_model.get_layer("res4w_out").output),  # for resnet100
            ("rpn_bbox", model.keras_model.get_layer("rpn_bbox").output),
            ("roi", model.keras_model.get_layer("ROI").output),
        ])

        # Input image (normalized)
        plt.figure("input_image")
        fig = plt.imshow(modellib.unmold_image(activations["input_image"][0], config))
        output_path = os.path.join(output_dir, "input_image_id_{}.png".format(image_id))
        print('Saving image {} to {}'.format(image_id, output_path))
        plt.savefig(output_path)

        # Backbone feature map
        fig = visualize.display_images(np.transpose(activations["res4w_out"][0, :, :, :8], [2, 0, 1]))
        output_path = os.path.join(output_dir, "feature_maps_id_{}.png".format(image_id))
        print('Saving image {} to {}'.format(image_id, output_path))
        fig.savefig(output_path)


def visualize_masks(model, dataset, config, image_ids, output_dir):
    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))
        # Run object detection
        results = model.detect([image], verbose=1)

        # Display results
        ax = get_ax(1)
        r = results[0]

        fig = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                          dataset.class_names, r['scores'], ax=ax,
                                          title="Predictions")
        # Now we can save it to a numpy array.
        '''
        data = fig2data(fig)
        remove_black_bars(data)
        plt.figure()
        plt.imshow(data)
        plt.savefig('/workspace/test.png')
        '''
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)
        output_path = os.path.join(output_dir, "instance_masks_id_{}.png".format(image_id))
        print('Saving image {} to {}'.format(image_id, output_path))
        fig.savefig(output_path)

        img = trim(trim(trim(Image.open(output_path))))

        img.save(output_path)


def compare_instance_masks(model, dataset, config, image_ids, output_dir, iou_threshold=0.5, score_threshold=0.5,
                           show_box=False, show_mask=True, ax=None, title=None):
    class_names = dataset.class_names

    for image_id in image_ids:
        image_id = int(image_id)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))
        # Run object detection
        results = model.detect([image], verbose=1)[0]
        pred_bbox = results['rois']
        pred_class_id = results['class_ids']
        pred_score = results['scores']
        pred_mask = results['masks']
        """Display ground truth and prediction instances on the same image."""
        # Match predictions to ground truth
        gt_match, pred_match, overlaps = utils.compute_matches(
            gt_bbox, gt_class_id, gt_mask,
            pred_bbox, pred_class_id, pred_score, pred_mask,
            iou_threshold=iou_threshold, score_threshold=score_threshold)

        # Ground truth = green. Predictions = red
        colors = [(0.686, 0.35, 0.18, .5)] * len(gt_match) \
                 + [(0.471, 0.558, 0.28, 1)] * len(pred_match)

        # Iterate over the GT and extract difference images between prediction and groundtruth.
        gt_mask_zeros = np.zeros_like(gt_mask)
        pred_mask_zeros = np.zeros_like(pred_mask)
        for gt_index in range(len(gt_match)):
            pred_index = int(gt_match[gt_index])
            # There was a match between prediction and ground truth
            if pred_index != -1:
                gt_inst_mask = gt_mask[:, :, gt_index]
                gt_mask_zeros[:, :, gt_index] = gt_inst_mask
                pred_inst_mask = pred_mask[:, :, pred_index]
                diff = np.abs(gt_inst_mask.astype(np.int) - pred_inst_mask.astype(np.int))
                pred_mask_zeros[:, :, pred_index] = diff
            else:
                gt_mask_zeros[:, :, gt_index] = np.zeros_like(gt_mask[:, :, gt_index])
        gt_mask = gt_mask_zeros
        pred_mask = pred_mask_zeros

        # Concatenate GT and predictions -> don't concatenate them but determine the respective differences
        class_ids = np.concatenate([gt_class_id, pred_class_id])
        scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
        boxes = np.concatenate([gt_bbox, pred_bbox])
        masks = np.concatenate([gt_mask, pred_mask], axis=-1)

        # Captions per instance show score/IoU
        captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
            pred_score[i],
            (overlaps[i, int(pred_match[i])]
             if pred_match[i] > -1 else overlaps[i].max()))
            for i in range(len(pred_match))]
        # Set title if not provided
        title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
        # Display
        fig = visualize.display_instances(
            image,
            boxes, masks, class_ids,
            class_names, scores, ax=ax,
            show_bbox=show_box, show_mask=show_mask,
            colors=colors, captions=None,
            title=title)
        output_path = os.path.join(output_dir, "instance_masks_diff_to_gt_id_{}.png".format(image_id))
        print('Saving image {} to {}'.format(image_id, output_path))
        fig.savefig(output_path)


def main():
    # Convert image_ids to ints
    args.image_ids = [int(image_id) for image_id in args.image_ids]
    # Load data
    config, dataset = prepare_coco(args.coco_dir)
    # Load model from checkpoint
    assert ".h5" in args.model_checkpoint, "Please specify a checkpoint file."
    model = load_model(args.model_checkpoint, config)
    assert os.path.exists(args.output_dir), "Output directory does not exist."

    if args.action == "feature_maps":
        visualize_feature_map(model, dataset, config, args.image_ids, args.output_dir)
    elif args.action == "visualize_masks":
        visualize_masks(model, dataset, config, args.image_ids, args.output_dir)
    elif args.action == "compare_instance_masks":
        compare_instance_masks(model, dataset, config, args.image_ids, args.output_dir)
    else:
        raise ValueError("action not available")


if __name__ == "__main__":
    main()
