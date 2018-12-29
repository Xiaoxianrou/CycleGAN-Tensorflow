import tensorflow as tf
import numpy as np
import argparse
import os

from PIL import Image
from models import CycleGAN

DATASET_PATH = './datasets/vangogh2pokemon'

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--cycle_loss_lambda', type=float, default=10, help='Cycle Consistency Loss coefficient')
parser.add_argument('--instance_normalization', default=True, type=bool, help="Use instance norm instead of batch norm")
parser.add_argument('--batch_size', default=1, type=int, help="Batch size")
parser.add_argument('--image_size', default=128, type=int, help="Image size")
parser.add_argument('--epoch_num', default=200, type=int, help="Epoch number")
args = parser.parse_args()
print(args)


def read_image(path):
    try:
        img = Image.open(path)
        img = img.resize((args.image_size, args.image_size))
        img_m = np.array(img)
    except IOError:
        print('fail to load image!')

    if len(img_m.shape) != 3 or img_m.shape[2] != 3:
        print('Wrong image {} with shape {}'.format(path, img_m.shape))
        return None

    img_m = img_m.astype(np.float32) / 255.0
    return img_m


def read_images(base_dir):
    image_dataset = {}
    for dir_name in ['trainA', 'trainB', 'testA', 'testB']:
        images_dir = os.path.join(base_dir, dir_name)
        images = []
        for idx, file in enumerate(os.listdir(images_dir)):
            file_path = os.path.join(images_dir, file)
            image = read_image(file_path)
            if image is not None:
                images.append(image)
        image_dataset[dir_name] = images
    return image_dataset


def main():
    image_dataset = read_images(DATASET_PATH)
    trainA = image_dataset['trainA']
    trainB = image_dataset['trainB']


    summary_writer = tf.summary.FileWriter('./logs')
    gan = CycleGAN(args)

    with tf.Session() as sess:
        gan.train(sess, summary_writer, trainA, trainB)


if __name__ == "__main__":
    main()
