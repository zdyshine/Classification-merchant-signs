# -*- coding: utf-8 -*-
"""
@author: zdy
"""

'''
#########################################################
enhance image data, create more dogs pciture to classify.

# 裁剪和填充图片
"""
resize_image_with_crop_or_pad:
    input:原始图像、后面两个参数是调整后的目标图像的大小。
    说明：如果原始图像的尺寸大于目标图像，则函数自动截取原始图像中居中的部分
          如果目标图像大于原始图像的尺寸，则函数自动在原始图像的四周填充0背景
"""

# 通过比例调整图像大小
"""
central_crop:
    参数：原始图像、一个（0,1]的比例值
"""
with tf.Session() as sess:
    central_cropped = tf.image.central_crop(img_data,0.5)
    plt.imshow(central_cropped.eval())
    plt.show()

#########################################################
'''
import os
import random
import string
#import datetime
import tensorflow as tf
from itertools import islice

max_pic_num = 80 # define one class of dog total picture numbers.

def GetDataPath():
    ''' get origin picture path'''

    flags = tf.app.flags
    flags.DEFINE_string("data_path", "./data0/", "directory of dogs, for enhance data.")
    flags.DEFINE_string("enhance_data_path", "./data0/enhance_5/", "directory for store enhance data.")
#    flags.DEFINE_string("data_path", "/home/xsr-ai/Desktop/DetectDogs/classifydog", "directory of dogs, for enhance data.")
#    flags.DEFINE_string("enhance_data_path", "/home/xsr-ai/Desktop/DetectDogs/enhance/", "directory for store enhance data.")
    FLAGS = flags.FLAGS

    print("data path:%s" % FLAGS.data_path)
    print("enhance data path:%s" % FLAGS.enhance_data_path)

    return FLAGS.data_path,FLAGS.enhance_data_path

def RandEnhancePicture(picname, savepath):
    ''' random use one of [tf.image.flip_up_down, tf.image.flip_left_right, tf.image.random_brightness,
    tf.image.random_contrast, tf.image.random_hue, tf.image.random_saturation, tf.image.adjust_gamma] image Ops
    to enhance origin picture, and save to enhance data path'''

    filename, suffix = os.path.splitext(picname)  # get picture name
    filename = os.path.basename(filename) # get base picture name
    filename += "_"
    img = None

    tf.reset_default_graph()
    image = tf.read_file(picname) # read picture from gving path.
    image_decode_jpeg = tf.image.decode_jpeg(image)
    image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg, dtype=tf.float32) # convert image dtype to float



    rand = random.randint(1,7) # we only use 7 image Ops.
#    if rand == 1: # flip up down
#        image_flip_up_down = tf.image.flip_up_down(image_decode_jpeg)
#        image_flip_up_down = tf.image.convert_image_dtype(image_flip_up_down, dtype=tf.uint8)
#        img = tf.image.encode_jpeg(image_flip_up_down)
#        
#    if rand == 2: # flip left right
#        image_flip_left_right = tf.image.flip_left_right(image_decode_jpeg)
#        image_flip_left_right = tf.image.convert_image_dtype(image_flip_left_right, dtype=tf.uint8)
#        img = tf.image.encode_jpeg(image_flip_left_right)
#        
    if rand == 1: # random adjust brightness 在某范围随机调整图片亮度 
        image_random_brightness = tf.image.random_brightness(image_decode_jpeg, max_delta=0.5)
        image_random_brightness = tf.image.convert_image_dtype(image_random_brightness, dtype=tf.uint8)
        img = tf.image.encode_jpeg(image_random_brightness)
        
    if rand == 2: # random adjust contrast 在某范围随机调整图片对比度
        image_random_contrast = tf.image.random_contrast(image_decode_jpeg, 0.2, 2)# 0.5,3修改0.5,2（改-0.5到2试试）
        image_random_contrast = tf.image.convert_image_dtype(image_random_contrast, dtype=tf.uint8)
        img = tf.image.encode_jpeg(image_random_contrast)
        
    if rand == 3: # random adjust hue 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间
        image_random_hue = tf.image.random_hue(image_decode_jpeg, max_delta=0.5)
        image_random_hue = tf.image.convert_image_dtype(image_random_hue, dtype=tf.uint8)
        img = tf.image.encode_jpeg(image_random_hue)
        
    if rand == 4: # random adjust saturation 在某范围随机调整图片饱和度 
        image_random_saturation = tf.image.random_saturation(image_decode_jpeg, 0.5, 3)
        image_random_saturation = tf.image.convert_image_dtype(image_random_saturation, dtype=tf.uint8)
        img = tf.image.encode_jpeg(image_random_saturation)
        
    if rand == 5: # adjust gamma
        image_adjust_gamma = tf.image.adjust_gamma(image_decode_jpeg, gamma=6)
        image_adjust_gamma = tf.image.convert_image_dtype(image_adjust_gamma, dtype=tf.uint8)
        img = tf.image.encode_jpeg(image_adjust_gamma)


####增加######

    if rand == 6: # 自动插值
        image_resize_images =  tf.image.resize_images(image_decode_jpeg,[500,500],method=3) 
        image_resize_images = tf.image.convert_image_dtype(image_resize_images, dtype=tf.uint8)
        img = tf.image.encode_jpeg(image_resize_images)
        


    if rand == 7: # 自动中央截取
        image_resize_image_with_crop_or_pad = tf.image.resize_image_with_crop_or_pad(image_decode_jpeg,500,500) 
        image_resize_image_with_crop_or_pad = tf.image.convert_image_dtype(image_resize_image_with_crop_or_pad, dtype=tf.uint8)
        img = tf.image.encode_jpeg(image_resize_image_with_crop_or_pad)        


    # save image
    openfile = filename + "".join(random.sample(string.digits, 8)) + suffix  # random rename picture avoid conflict
    hd = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")
    with tf.Session() as sess:  # create tensorflow session
        img = sess.run(img)

    tf.get_default_graph().finalize()

    hd.write(img)
    hd.close()


def EnhancePictureAndSave(picname, savepath):
    ''' use one of [tf.image.flip_up_down, tf.image.flip_left_right, tf.image.random_brightness,
    tf.image.random_contrast, tf.image.random_hue, tf.image.random_saturation, tf.image.adjust_gamma] image Ops
    to enhance origin picture, and save to enhance data path'''

    tf.reset_default_graph()

    filename, suffix = os.path.splitext(picname)      # get picture path
    filename = os.path.basename(filename)            # get base picture name
    filename += "_"

    image = tf.read_file(picname)                # read picture from gving path.
    image_decode_jpeg = tf.image.decode_jpeg(image)
    image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg, dtype=tf.float32) # convert image dtype to float



    # flip up down1
#    image_flip_up_down = tf.image.flip_up_down(image_decode_jpeg)
#    image_flip_up_down = tf.image.convert_image_dtype(image_flip_up_down, dtype=tf.uint8)
#    image_flip_up_down = tf.image.encode_jpeg(image_flip_up_down)
#
#    # save image
#    openfile = filename +  "".join(random.sample(string.digits, 8)) + suffix # random rename picture avoid conflict
#    hd_up_down = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")
#
#
#
#
#    # flip left right2
#    image_flip_left_right = tf.image.flip_left_right(image_decode_jpeg)
#    image_flip_left_right = tf.image.convert_image_dtype(image_flip_left_right, dtype=tf.uint8)
#    image_flip_left_right = tf.image.encode_jpeg(image_flip_left_right)
#
#    # save image
#    openfile = filename + "".join(random.sample(string.digits, 8)) + suffix  # random rename picture avoid conflict
#    hd_left_right = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")




    # random adjust brightness3
    image_random_brightness = tf.image.random_brightness(image_decode_jpeg, max_delta=0.01)
    image_random_brightness = tf.image.convert_image_dtype(image_random_brightness, dtype=tf.uint8)
    image_random_brightness = tf.image.encode_jpeg(image_random_brightness)

    # save image
    openfile = filename + "".join(random.sample(string.digits, 8)) + suffix  # random rename picture avoid conflict
    hd_adj_brightness = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")




    # random adjust contrast4
    image_random_contrast = tf.image.random_contrast(image_decode_jpeg, 0.8, 1)
    image_random_contrast = tf.image.convert_image_dtype(image_random_contrast, dtype=tf.uint8)
    image_random_contrast = tf.image.encode_jpeg(image_random_contrast)

    # save image
    openfile = filename + "".join(random.sample(string.digits, 8)) + suffix  # random rename picture avoid conflict
    hd_adj_contrast = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")




    # random adjust hue5
    image_random_hue = tf.image.random_hue(image_decode_jpeg, max_delta=0.05)
    image_random_hue = tf.image.convert_image_dtype(image_random_hue, dtype=tf.uint8)
    image_random_hue = tf.image.encode_jpeg(image_random_hue)

    # save image
    openfile = filename + "".join(random.sample(string.digits, 8)) + suffix  # random rename picture avoid conflict
    hd_adj_hue = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")




    # random adjust saturation6
    image_random_saturation = tf.image.random_saturation(image_decode_jpeg, 0.7, 1)
    image_random_saturation = tf.image.convert_image_dtype(image_random_saturation, dtype=tf.uint8)
    image_random_saturation = tf.image.encode_jpeg(image_random_saturation)

    # save image
    openfile = filename + "".join(random.sample(string.digits, 8)) + suffix  # random rename picture avoid conflict
    hd_adj_saturation = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")




    # adjust gamma7
    image_adjust_gamma = tf.image.adjust_gamma(image_decode_jpeg, gamma=2)
    image_adjust_gamma = tf.image.convert_image_dtype(image_adjust_gamma, dtype=tf.uint8)
    image_adjust_gamma = tf.image.encode_jpeg(image_adjust_gamma)

    # save image
    openfile = filename + "".join(random.sample(string.digits, 8)) + suffix  # random rename picture avoid conflict
    hd_adj_gamma = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")


###增加####

    # adjust gamma  888
    image_resize_images = tf.image.resize_images(image_decode_jpeg,[500,500],method=3)
    image_resize_images = tf.image.convert_image_dtype(image_resize_images, dtype=tf.uint8)
    image_resize_images = tf.image.encode_jpeg(image_resize_images)

    # save image
    openfile = filename + "".join(random.sample(string.digits, 8)) + suffix  # random rename picture avoid conflict
    hd_adj_8 = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")
    
    
    
    # adjust gamma  999
    image_resize_image_with_crop_or_pad = tf.image.resize_image_with_crop_or_pad(image_decode_jpeg,500,500 )
    image_resize_image_with_crop_or_pad = tf.image.convert_image_dtype(image_resize_image_with_crop_or_pad, dtype=tf.uint8)
    image_resize_image_with_crop_or_pad = tf.image.encode_jpeg(image_resize_image_with_crop_or_pad)

    # save image
    openfile = filename + "".join(random.sample(string.digits, 8)) + suffix  # random rename picture avoid conflict
    hd_adj_9 = tf.gfile.FastGFile(os.path.join(savepath, openfile), "w")
    
    
    

    with tf.Session() as sess:  # create tensorflow session
#        img_up_down, img_left_right, img_brightness, img_contrast, img_hue, img_saturation, img_gamma \
#            = sess.run([image_flip_up_down, image_flip_left_right, image_random_brightness, image_random_contrast,
#                        image_random_hue, image_random_saturation, image_adjust_gamma])
 
        img_brightness, img_contrast, img_hue, img_saturation, img_gamma,img_8,img_9 \
            = sess.run([ image_random_brightness, image_random_contrast,
                        image_random_hue, image_random_saturation, image_adjust_gamma,image_resize_images,image_resize_image_with_crop_or_pad])    
    
    
    
    tf.get_default_graph().finalize()
    
#    hd_up_down.write(img_up_down)
#    hd_up_down.close()
#    
#    hd_left_right.write(img_left_right)
#    hd_left_right.close()
    
    hd_adj_brightness.write(img_brightness)
    hd_adj_brightness.close()
    
    hd_adj_contrast.write(img_contrast)
    hd_adj_contrast.close()
    
    hd_adj_hue.write(img_hue)
    hd_adj_hue.close()
    
    hd_adj_saturation.write(img_saturation)
    hd_adj_saturation.close()
    
    hd_adj_gamma.write(img_gamma)
    hd_adj_gamma.close()

###增加####
    hd_adj_8.write(img_8)
    hd_adj_8.close()

    hd_adj_9.write(img_9)
    hd_adj_9.close()

def EnhanceData():
    '''enhance picture to max count 600 handling'''
    data_path,enhance_path = GetDataPath()

    if tf.gfile.Exists(enhance_path): # make enhance data directory
        tf.gfile.DeleteRecursively(enhance_path)

    walk = tf.gfile.Walk(data_path)
    walk = islice(walk, 1, None) # skip parent directory
    
    for info in walk:
        basedir = os.path.basename(info[0])
        tf.gfile.MakeDirs(os.path.join(enhance_path, basedir))

        for pic in info[2]: # copy origin picture to save path
            picname = os.path.join(info[0], pic)  # join path and picname
            tf.gfile.Copy(picname, os.path.join(enhance_path, basedir, os.path.basename(picname)))
            remaincount = max_pic_num - len(info[2])

        # picture total nums is 600, that will make every picture enhance enhance_times_per_pic
        if len(info[2]) <= (remaincount / 7):
            for pic in info[2]:
                picname = os.path.join(info[0], pic) # join path and picname
                EnhancePictureAndSave(picname, os.path.join(enhance_path, basedir))
                remaincount -= 7 # every time enhance picture will increase 7 frame.

        for index in range(remaincount):
            rand = random.randint(0, len(info[2])-1)
            picname = os.path.join(info[0], info[2][rand])  # join path and picname
            RandEnhancePicture(picname, os.path.join(enhance_path, basedir))
            #sess.close() # close tensorflow session

if __name__ == "__main__":
    print("begin to enhance picture data!!!")
    EnhanceData()
    print("end of enhance picture data, good luck!!!")
    
    
    
    
    
    
    
    