# Autonomous Vehicle Object Detector
Built an Object Detector using Darknet and YOLOv4 to detect Traffic Signs, Traffic Lights, and Cars

## Introduction 
With autonomous vehicles taking off in the past several years, I want to explore one aspect that was necessary to create a good autonomous vehicle system, object tracking. So what is object tracking exactly? Best way to describe it is the "eyes of car" 

![Tesla](/images/tesla.gif) 

## Process
So first I would actually understand how to go about this. I was referred to an open source Google Colab notebook by the AI Guy and used that to help springboard my modeling. To actually model this I would be using the Darknet open-source neural network and YOLO object detector to create my model. To explain the power of YOLO, a traditional object tracker utilizes a sliding window. 

## Traditional Object Detection
![sliding_window](/images/sliding_window.gif) 
[Source](https://towardsdatascience.com/how-do-self-driving-cars-see-13054aee2503)
    
As you can observe, the window goes over every part of the image until it detects the actual object, in this case the car. There are two boxes because one is the Ground Truth Box where before it was outlined exactly where the object was located in the picture and the other is the Predicted Box which is where the model predicts the object is. This is used to calculate the Intersection of Union which (IoU) calculates the Mean Average Precision (MAP) and something we'll touch upon later. Overall, this process is very computer intensive and very inefficient for object detection. In terms of autonomous vehicles, you would not want your car to not recognize the Stop Sign until 15 seconds later and then all of a sudden stop. It needs to be instantaneous 

## YOLO

![yoloimage](/images/yoloimage.png)
[Source](https://towardsdatascience.com/how-do-self-driving-cars-see-13054aee2503)

Introduce You Only Look Once or YOLO for short. Joseph Redmon is a computer wiz that created YOLO back in 2015, and he also maintains the Darknet neural network. (Link to YouTube TED Talk). What happens in YOLO differently than the above is there are 3-steps so the model can just go over an image/video frame one time. Grid segmentation, Classification, and Image Localization. Grid segmentation breaks down the picture into evenly sized gridblocks so every part of the picture is accounted for. Then the model will identify the different classes of the image, in this case "dog", "bike", and "truck". Finally, the objects are located using bounding boxes which locate where the objects are within the image, hence the name Image Localization. Putting all that together, you have your model that has successfully identified a dog, bike, and truck and the locations of all of them within an image. We'll get to the efficiency aspect of it later on.

## Image Collection and Pretrained Weights
First, we need to collect some images. I collected all my images of Cars, Traffic Lights, Traffic Signs, and Stop Signs from Google Open Images. Before creating the model, I downloaded YOLOv4 Pretrained weights which are trained on Microsoft's COCO Dataset of 80 classes. In those 80 classes you have Cars, Traffic Lights, and Stop Signs. So it would be wise to take advantage of it to increase the mAP (Mean Accuracy Precision). Unfortunately, there are no pre-trained weights for Traffic Signs hence why it is predicted that the mAP score for it will be lower than the others. 

## Model Creation and Testing
Now that the model is created, I ran my videos through it and VOILA! 
![yolomodel](/images/yolomodel.gif)

As you can see it picks up all the Traffic Lights, Cars, and most importantly Traffic Signs! Great, first iteration of the model and it works well! 

## Results: Mean Accuracy Precision 
If you recall from above, we spoke about IoU (Intersection of Union) and how that would impact our mAP (Mean Accuracy Precision). To breakdown the IoU once again, please observe the picture of the nice kitten below: 

![cat](/images/cat.png)
  
As you can observe, there is a Ground-Truth Bounding Box and Predicted Box. The Ground-Truth Bounding Box is drawn by the user manually before the model is built to indicate exactly where the object is within the picture. The Predicted Box is the model determining where it "thinks" is the object. The greater the intersection between the two or where they overlap the greater the Average Precision (AP) score will be. An AP is calculated for every single object class in each image and then all the scores are averaged to determine the mAP score, which ultimately decides how well your model is doing.

![IoU](/images/iou.png)

![example](/images/example.png)

Please see below for individual class mAPs and overall mAP:
List: 
- Cars: 80.70%
- Stop Signs: 98.20%
- Traffic Lights: 75.06%
- Traffic Signs: 42.49% 

Overall mAP: 74.11%
As expected the Traffic Signs have the lowest mAP since it did not have any pretrained weights to train on. I would like to improve this score for future iterations because no one wants to get into an Autonomous Vehicle where it can only predict every 1 out of 2 times if it is a traffic sign or not. 

## Future Work
I would like to continue working on this project and deploy it on mobile applications. There is an Android folder I deployed in the open source repo using Android Studio. It was deployed with the YOLOv4 pretrained weights and as seen below, it can detect cars, trucks, and traffic lights in real time. I customized the app name to "DashKam" and changed the layout of the app, besides that all the code is from the AI Guys repository. Ideally, I would like to deploy this mobile app with my own YOLOv4 custom trained model in a TensorFlow Lite format. 

![android](/images/android.gif)

## Ethical Dilemma
Joseph Redmon created YOLO to push the boundaries of object detection. With his model we are able to build models instantaneously detect objects. However, he was approached by a military personnel that told him they use his model to track vehicles and people, which ultimately can lead to drone strikes. Redmon was so horrified hearing this that at the beginning of 2020, he proclaimed he would not be doing anymore work with computer vision (this includes object detection) going forward. What he created was so beautiful and magnificent but it shows that if it falls into the wrong hands it can lead to devastating consequences. We as Data Scientists have a responsibility to be ethical in our work for the sake of society.

## Special Thanks
- [the AI Guy](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial)
- [hunglc007](https://github.com/hunglc007/tensorflow-yolov4-tflite)
