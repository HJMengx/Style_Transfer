import argparse
import cv2
import tensorflow as tf
from neural_style_transfer import *

def arg_parse():

    parser = argparse.ArgumentParser(description='Style_transfer')

    parser.add_argument("--style", dest = 'style', help =
                        "Image / Directory containing images of Style,if you don`t select that will select default image",
                        default='sandstone.jpg', type = str)
    parser.add_argument("--content", dest = 'content', help =
                        "Image / Directory containing images of content", type = str)
    parser.add_argument("--dest", dest = 'dest', help =
                        "Image / Directory to store transfer_image to",
                        default = "output", type = str)

    return parser.parse_args()

args = arg_parse()
style_image = args.style
content_image = args.content
destination = args.dest

if content_image is None:
    raise  Exception("please select the content image")

content_image = cv2.imread('cat.jpg')[:,:,::-1]
content_image = reshape_and_normalize_image(content_image)
style_image = cv2.imread("sandstone.jpg")[:,:,::-1]
style_image = reshape_and_normalize_image(style_image)
with tf.Graph().as_default():
    with tf.Session() as sess:
        g_image = generate(sess,content_image,style_image)
        sess.close()
save_image('generate.jpg',g_image)


