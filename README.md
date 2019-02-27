# Classification-merchant-signs
2018 Baidu merchant signboard classification and testing contest：2018百度商家招牌的分类与检测大赛

# Overview
This is a summary of my participation in the classification and testing competition of 2018 Baidu merchant signs. Competition homepage: [百度]（https://dianshi.baidu.com/competition/17/rule）。  <br>
The project runs on win10 + anaconada. You can also use other environments to run. <br>
The program uses the InceptionV3 model for fine-tuning, and the final classification accuracy is 0.995. <br>
I also tried to use resnet50, vgg16, Xception...they can also get a nice result. <br>

# Network structure change
The final fully connected layer and category output layer of the network has changed according to the number of categories of actual classified items. You can also try to make different changes.

# Installation requirements
python = 3.6.0 <br>
tensorflow >= 1.7.0 <br>
keras > = 2.1.3 <br>
argparse <br>
matplotlib <br>

# Project file
data_pre.py : Divide the training set into a training set and a validation set. <br>
data_arguement.py : Data enhancement,The default is to enhance 1 to 8 images. <br>
finetune_model.py : Use this script to train. <br>
finetune_model_test.py : Use this script to test. <br>

# Dataset file directory
Before enhancement <br>
  | datasets  <br>
     | test
       |image1.jpg
        image2.jpg
        ... <br>
     | train <br>
       |image1.jpg
        image2.jpg
        ... <br>
     | test.txt <br>
     | train.txt <br>
     
After enhancement <br>
  | datasets <br>
     | test <br>
       |image1.jpg
        image2.jpg
        ... <br>
     | train <br>
       | 1   <br>
        |image1.jpg
         image2.jpg
         ... <br>
       | 2  <br>
        |image1.jpg
         image2.jpg
         ...  
       ...
     | valid <br>
       | 1  
        |image1.jpg
         image2.jpg
         ...
       | 2  <br>
        |image1.jpg
         image2.jpg
         ...  <br>
       ... <br>
     | test.txt <br>
     | train.txt <br>
     
Txt file format : image name + label.

# Match Results
[image] (
        Classification-merchant-signs/image/result.jpg
      )
