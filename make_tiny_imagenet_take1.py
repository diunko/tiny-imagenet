import argparse, os, os.path, glob, random, sys, json
from collections import defaultdict
from lxml import objectify

from scipy.misc import imread, imsave, imresize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

train_anns_path = 'imagenet/annotations'
train_image_dir = 'imagenet/images'


def parse_xml_file(filename):
    with open(filename, 'r') as f:
        xml = f.read()
    ann = objectify.fromstring(xml)
    img_filename = '%s.JPEG' % ann.filename
    bbox = ann.object.bndbox
    bbox = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
    bbox = [int(x) for x in bbox]
    name = str(ann.object.name)
    return img_filename, bbox, name


def resize_image(img, size, bbox=None, crop=True, show=False):
    """
    Resize an image and its bounding box to a square.

    img - A numpy array with pixel data for the image to resize.
    size - Integer giving the height and width of the resized image.
    bbox - Optionally, a list [xmin, ymin, xmax, ymax] giving the coordinates
           of a bounding box in the original image.
    crop - If true, center crop the original image before resizing; this avoids
           distortion in images with nonunit aspect ratio, but may also crop out
           part of the object.
    show - If true, show the original and resized image and bounding box.

    Returns:
    If bbox was passed: (img_resized, bbox_resized)
    otherwise: img_resized
    """

    def draw_rect(coords):
        width = coords[2] - coords[0]
        height = coords[3] - coords[1]
        rect = Rectangle((coords[0], coords[1]), width, height,
                         fill=False, linewidth=2.0, ec='green')
        plt.gca().add_patch(rect)

    img_resized = img
    if bbox is not None:
        bbox_resized = [x for x in bbox]
    if crop:
        h, w = img.shape[0], img.shape[1]
        if h > w:
            h0 = (h - w) / 2
            if bbox is not None:
                bbox_resized[1] -= h0
                bbox_resized[3] -= h0
            img_resized = img[h0:h0 + w, :]
        elif w > h:
            w0 = (w - h) / 2
            if bbox is not None:
                bbox_resized[0] -= w0
                bbox_resized[2] -= w0
            img_resized = img[:, w0:w0 + h]

    if bbox is not None:
        h_ratio = float(size) / img_resized.shape[0]
        w_ratio = float(size) / img_resized.shape[1]
        ratios = [w_ratio, h_ratio, w_ratio, h_ratio]
        bbox_resized = [int(1 + r * (x - 1)) for x, r in zip(bbox_resized, ratios)]
        bbox_resized = np.clip(bbox_resized, 0, size - 1)
    img_resized = imresize(img_resized, (size, size))

    if show:
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        if bbox is not None:
            draw_rect(bbox)
        plt.subplot(1, 2, 2)
        plt.imshow(img_resized)
        if bbox is not None:
            draw_rect(bbox_resized)
        plt.show()

    if bbox is None:
        return img_resized
    else:
        return img_resized, bbox_resized


def parse_xml_file(filename):
    with open(filename, 'r') as f:
        xml = f.read()
    ann = objectify.fromstring(xml)
    img_filename = '%s.JPEG' % ann.filename
    bbox = ann.object.bndbox
    bbox = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
    bbox = [int(x) for x in bbox]
    name = str(ann.object.name)
    return img_filename, bbox, name


def write_data_in_synset_folders(part_data, part, out_dir, image_size):
    part_dir = os.path.join(out_dir, part)
    os.mkdir(part_dir)
    num_wnids = len(part_data)
    for i, (wnid, wnid_data) in enumerate(part_data.iteritems()):
        print >> sys.stderr, 'Writing images for synset %d / %d of %s' % (i + 1, num_wnids, part)
        wnid_dir = os.path.join(part_dir, wnid)
        os.mkdir(wnid_dir)
        image_dir = os.path.join(wnid_dir, 'images')
        os.mkdir(image_dir)
        boxes_filename = os.path.join(wnid_dir, '%s_boxes.txt' % wnid)
        boxes_file = open(boxes_filename, 'w')
        for i, (img_filename, bbox) in enumerate(wnid_data):
            out_img_filename = '%s_%d.JPEG' % (wnid, i)
            print >> sys.stderr, img_filename
            full_out_img_filename = os.path.join(image_dir, out_img_filename)
            img = imread(img_filename)
            img_resized, bbox_resized = resize_image(img, image_size, bbox)
            imsave(full_out_img_filename, img_resized)
            boxes_file.write('%s\t%d\t%d\t%d\t%d\n' % (out_img_filename,
                                                       bbox_resized[0], bbox_resized[1], bbox_resized[2],
                                                       bbox_resized[3]))
        boxes_file.close()


def write_data_in_one_folder(part_data, part, out_dir, image_size):
    part_dir = os.path.join(out_dir, part)
    os.mkdir(part_dir)

    # First flatten the part data so we can shuffle it
    part_data_flat = []
    for wnid, wnid_data in part_data.iteritems():
        for (img_filename, bbox) in wnid_data:
            part_data_flat.append((wnid, img_filename, bbox))

    random.shuffle(part_data_flat)
    image_dir = os.path.join(part_dir, 'images')
    os.mkdir(image_dir)

    annotations_filename = os.path.join(part_dir, '%s_annotations.txt' % part)
    annotations_file = open(annotations_filename, 'w')
    for i, (wnid, img_filename, bbox) in enumerate(part_data_flat):
        if i % 100 == 0:
            print >> sys.stderr, 'Finished writing %d / %d %s images' % (i, len(part_data_flat), part)
        out_img_filename = '%s_%s.JPEG' % (part, i)
        full_out_img_filename = os.path.join(image_dir, out_img_filename)
        img = imread(img_filename)
        img_resized, bbox_resized = resize_image(img, image_size, bbox)
        imsave(full_out_img_filename, img_resized)
        annotations_file.write('%s\t%s\t%d\t%d\t%d\t%d\n' % (
            out_img_filename, wnid,
            bbox_resized[0], bbox_resized[1], bbox_resized[2], bbox_resized[3]))
    annotations_file.close()


def make_tiny_imagenet(wnids, num_train, num_val, out_dir, image_size=50, seed=42):
    if os.path.isdir(out_dir):
        print >> sys.stderr, 'Output directory already exists'
        return

    # dataset['train']['n123'][0] = (filename, (xmin, ymin, xmax, xmax))
    # gives one example of an image and bbox for synset n123 of the training subset
    dataset = defaultdict(lambda: defaultdict(list))

    random.seed(seed)

    for i, wnid in enumerate(wnids):
        print >> sys.stderr, 'Choosing train and val images for synset %s %d / %d' % (wnid, i + 1, len(wnids))

        # TinyImagenet train and val images come from ILSVRC-2012 train images
        train_synset_dir = os.path.join(train_anns_path, wnid)
        orig_train_bbox_files = os.listdir(train_synset_dir)
        orig_train_bbox_files = {os.path.join(train_synset_dir, x) for x in orig_train_bbox_files}

        print >> sys.stderr, "train", num_train, "of", len(orig_train_bbox_files)
        train_bbox_files = random.sample(orig_train_bbox_files, num_train)
        orig_train_bbox_files -= set(train_bbox_files)

        print >> sys.stderr, "val", num_val, "of", len(orig_train_bbox_files)
        val_bbox_files = random.sample(orig_train_bbox_files, num_val)

        for bbox_file in train_bbox_files:
            img_filename, bbox, _ = parse_xml_file(bbox_file)
            img_filename = os.path.join(train_image_dir, img_filename)
            dataset['train'][wnid].append((img_filename, bbox))

        for bbox_file in val_bbox_files:
            img_filename, bbox, _ = parse_xml_file(bbox_file)
            img_filename = os.path.join(train_image_dir, img_filename)
            dataset['val'][wnid].append((img_filename, bbox))

    # Now that we have selected the images for the dataset, we need to actually
    # create it on disk
    os.mkdir(out_dir)
    write_data_in_synset_folders(dataset['train'], 'train', out_dir, image_size)
    write_data_in_one_folder(dataset['val'], 'val', out_dir, image_size)


parser = argparse.ArgumentParser()
parser.add_argument('--wnid_file', type=argparse.FileType('r'))
parser.add_argument('--num_train', type=int, default=100)
parser.add_argument('--num_val', type=int, default=100)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out_dir')
args = parser.parse_args()

if __name__ == '__main__':
    wnids = [line.strip() for line in args.wnid_file]
    print >> sys.stderr, len(wnids)
    # wnids = ['n02108089', 'n09428293', 'n02113799']
    make_tiny_imagenet(wnids, args.num_train, args.num_val, args.out_dir,
                       image_size=args.image_size)
    sys.exit(0)

    train_synsets = os.listdir(train_anns_path)

    get_synset_stats()
    sys.exit(0)

#  python make_tiny_imagenet_take1.py \
#             --wnid_file 200_wnids.txt \
#             --num_train 500 \
#             --num_val 50
#             --out_dir output
#             --image_size 224
