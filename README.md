# Autonomous Vehicle Object Detector
With autonomous vehicles taking off in the past several years, I wanted to explore one aspect that was necessary to create a good autonomous vehicle system, object detection. So what is it exactly? Best way to describe it is the "eyes of car". As you see below, the vehicles' cameras are feeding the autonomous system what objects it is seeing. In this case it is a person, pets, and other vehicles. The autonomous system then uses the information to make a decision to turn left, go straight, and etc. Let's see if we can replicate this Object Detector using Darknet and YOLOv4 to detect traffic signs, traffic lights, and other vehicles.

![Tesla](/images/tesla.gif) 

## Process
First I wanted to understand how to approach this project. I was referred to an open source Google Colab notebook by the AI Guy and used that to help springboard my modeling. To create the model I would be using the Darknet open-source neural network and YOLO object detector. To explain the power of YOLO, lets first compare it to a traditional object tracker which utilizes a sliding window.

## Traditional Object Detection
![sliding_window](/images/sliding_window.gif)    

- [Source](https://towardsdatascience.com/how-do-self-driving-cars-see-13054aee2503)
    
As you can observe, the window goes over every part of the image until it detects the actual object, in this case the car. There are two boxes: a Ground Truth Box which was was placed in the picture manually prior to modeling and the other is the Predicted Box which is where the model predicts where the object is. These two boxes are used to calculate the Intersection of Union (IoU) which calculates the Mean Average Precision (mAP) and something we'll touch upon later in the article. Overall, this process is very computer intensive and inefficient for object detection. In terms of autonomous vehicles, you would not want your car to not recognize a "stop sign" 15 seconds later and then all of a sudden stop. It needs to be instantaneous.

## YOLO

![yoloimage](/images/yoloimage.png)      

- [Source](https://towardsdatascience.com/how-do-self-driving-cars-see-13054aee2503)

Now introducing You Only Look Once or YOLO for short. Joseph Redmon is a computer wiz that created YOLO back in 2015, and he also maintains the Darknet neural network (check out his TED Talk [here](https://www.youtube.com/watch?v=XS2UWYuh5u0). What happens in YOLO differently than the above are these three steps: Grid segmentation, Classification, and Image Localization. In this method the model needs to go over an image/video frame one time. Grid segmentation breaks down the picture into evenly sized grid-blocks so every part of the picture is accounted for. Then the model will identify the different classes of the image, in this case dog, bike, and truck. Finally, the objects are located using bounding boxes, as mentioned above, which locate where the objects are within the image, hence the name Image Localization. Putting all that together, you have your model that has successfully identified a dog, bike, and truck and the locations of all of them within an image.

## Image Collection and Pretrained Weights
Before we start building our model, we need to collect some images. I collected all my images of cars, traffic lights, traffic signs, and stop signs from Google Open Images. Before creating the model, I downloaded YOLOv4 Pre-trained weights which are trained on Microsoft's COCO Dataset of 80 classes. In those 80 classes you have cars, traffic lights, and stop signs. So it would be wise to take advantage of it to increase the mAP (Mean Accuracy Precision). Unfortunately, there are no pre-trained weights for traffic signs hence why it is predicted that the mAP score for it will be lower than the others.

## Model Creation and Testing
Now that the model is created, I ran my videos through it and VOILA! 
![yolomodel](/images/yolomodel.gif)

As you can see it picks up all the traffic lights, cars, and most importantly traffic signs! Good, first iteration of the model!

## Results: Mean Accuracy Precision 
If you recall from above, I mentioned Intersection of Union (IoU) and how that would impact our Mean Accuracy Precision (mAP). To breakdown the IoU once again, please observe the picture of the nice kitten below.

![cat](/images/cat.png)
- [Source](https://blog.paperspace.com/mean-average-precision/)  

As you can observe, there is a Ground-Truth Bounding Box and Predicted Box. The Ground-Truth Bounding Box is drawn manually before the model is built to indicate exactly where the object is within the picture. The Predicted Box is the model determining where it "thinks" the object is. The greater the intersection between the two bounding boxes the greater the Average Precision (AP) score. An AP is calculated for every single object class in each image and then all the scores are averaged to determine the mAP score, which ultimately decides how well your model is doing.

![IoU](/images/iou.png)
- [Source](https://blog.paperspace.com/mean-average-precision/)

Please see below for a more visual representation. As you can see the yellow box the "Predicted Box" is 90% overlapping with the Ground-Truth Bound Box.   

![example](/images/example.png)
- [Source](https://blog.paperspace.com/mean-average-precision/)

Finally, see below for individual class mAPs and overall mAP for the model.
- Cars: 80.70%
- Stop Signs: 98.20%
- Traffic Lights: 75.06%
- Traffic Signs: 42.49%
- Overall mAP: 74.11%

As expected the Traffic Signs have the lowest mAP since it did not have any pre-trained weights to train on. I would like to improve this score for future iterations because no one wants to get into an Autonomous Vehicle where it can only predict every 1 out of 2 times if it is a traffic sign or not. However, an overall mAP of 74.11%, not bad for now.

## Future Work
I would ideally like to deploy this model on mobile applications as a dashcam app. There is an Android folder I deployed from this open source repo using Android Studio. It was deployed with the YOLOv4 pre-trained weights and as seen below, it can detect cars, trucks, and traffic lights in real time. I customized the app name to "DashKam" and changed the layout, besides that all the code remained the same. Ideally, I would like to deploy this mobile app with my own YOLOv4 custom trained model in a TensorFlow Lite format.

![android](/images/android.gif)

## Ethical Dilemma
Joseph Redmon created YOLO to push the boundaries of object detection. With his model we are able to detect objects instantaneously. However, he was approached by a military personnel that informed him they use his model to track vehicles and people, which ultimately can lead to drone strikes. Redmon was so horrified hearing this, that at the beginning of 2020, he proclaimed he would not be doing anymore work with computer vision (which object detection falls under) going forward. What he created was so beautiful and magnificent but it shows that if it falls into the wrong hands it can lead to devastating consequences. For that, I strongly believe we need to hold onto the notion that we in the Data Science community have a responsibility to be ethical in our work for the sake of society.

## Special Thanks
- [the AI Guy](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial)
- [hunglc007](https://github.com/hunglc007/tensorflow-yolov4-tflite)
