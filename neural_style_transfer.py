import numpy as np
import keras
import keras.layers as layers
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D,Activation
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
import tensorflow as tf
import cv2
import sys
from nst_utils import *

# Model直接使用VGG,Inception,resnet进行迁移学习

def build_model():
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    print(model)
    return model

def content_loss(a_C,a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.transpose(a_C)
    a_G_unrolled = tf.transpose(a_G)

    # compute the cost with tensorflow (≈1 line)
    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.pow((a_G_unrolled - a_C_unrolled), 2))
    return J_content

def style_matrix(activation):
    return tf.matmul(activation,tf.transpose(activation))

def style_layer_loss(g,s):
    m,nH,nW,nC = g.get_shape().as_list()
    # 先reshape
    s = tf.transpose(tf.reshape(s, [nH*nW, nC]))
    g = tf.transpose(tf.reshape(g, [nH*nW, nC]))

    g_matrix = style_matrix(g)
    s_matrix = style_matrix(s)

    j_style_layer_loss = 1./(4 * nC**2 * (nH*nW)**2) * tf.reduce_sum(tf.pow((s_matrix,g_matrix),2))

    J_style_layer = (1. / (4 * nC ** 2 * (nH * nW) ** 2)) * tf.reduce_sum(tf.pow((s_matrix - g_matrix), 2))
    return J_style_layer

def style_loss(sess,layers,model):
    s_loss = 0
    for layer,coeff in layers:
        # 取出每一层的输出
        out = model[layer]
        a_s = sess.run(out)
        a_g = out
        s_loss += (coeff * style_layer_loss(a_g,a_s))
    return s_loss

def loss(J_content,J_style,alpha = 10,beta = 40):
    '''
    整个loss的计算是通过以下几个步骤(C:Content,G:Generate,S:Style):
    1. J(C,G) = (1 / (4 * nH * nW * nC)) * (sum((aC - aG) ** 2))  # 可以选定中间的某一卷积层作为输出,使用不同的卷积层会有不同的效果
    2. J(S,G)(layer) = (1 / (4 * nH ** 2 * (nW * nC) ** 2) * sum((s_matrix-g_matrix) ** 2) # 某一层的计算,如果是多层需要求和每一层和系数的积
    3. J(G) = αJcontent(C,G)+βJstyle(S,G)
    :return:
    '''
    return alpha * J_content + beta * J_style

# generate the style_transfer_image
def generate(sess,content_img,style_img,epoches = 140):
    generate_image = generate_noise_image(content_image)
    # 计算loss
    # 读取图片
    if type(content_img) != type(np.array([])):
        content_img = cv2.imread(content_img)[:,:,::-1]
    if type(style_img) != type(np.array([])):
        style_img = cv2.imread(style_img)[:,:,::-1]
    if content_img.shape != style_img.shape:
        style_img = cv2.resize(style_img,content_img.shape)
    # 获取模型
    model = build_model()
    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(2.0)
    output = model['conv4_2']
    # 计算Content_loss
    sess.run(model['input'].assign(content_img))
    a_C = sess.run(output)
    a_G = output
    # print(a_C,a_G)
    c_loss = content_loss(a_C,a_G)
    # layers
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]
    # 计算 style_loss
    sess.run(model['input'].assign(style_image))
    j_style = style_loss(sess,STYLE_LAYERS,model)
    # total_loss
    t_loss = loss(c_loss,j_style)
    # define train_step (1 line)
    train_step = optimizer.minimize(t_loss)
    np.fft.fftshift()
    # 得到Model之后初始化
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(generate_image))
    for i in range(epoches):
        # Compute gradients of `loss` for the variables in `var_list`.
        # 如果设置 tf.Variable(trainable=False),就可以不参与训练
        # 在这一步,反向传播会将两次都求偏导 源码:Tensorflow.training.optimizer code in:469开始
        sess.run(train_step)
        generate_image = sess.run(model['input'])
        if i % 20 == 0:
            t,c,s = sess.run([t_loss,c_loss,j_style])
            print("epoche:",i)
            print("total cost = " + str(t))
            print("content cost = " + str(c))
            print("style cost = " + str(s))
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generate_image)
    return generate_image

if __name__ == '__main__':
    content_image = cv2.imread('cat.jpg')[:,:,::-1]
    content_image = reshape_and_normalize_image(content_image)
    style_image = cv2.imread("sandstone.jpg")[:,:,::-1]
    style_image = reshape_and_normalize_image(style_image)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            g_image = generate(sess,content_image,style_image)
            sess.close()
    save_image('generate.jpg',g_image)