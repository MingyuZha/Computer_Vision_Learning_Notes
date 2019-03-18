# Computer Vision Learning Notes
## SIFT
### Why SIFT?
当图片存在不同的尺度(scales)以及旋转(rotations)时，简单的corner detector无法取得很好的效果，这时我们就需要使用**Scale Invariant Feature Transform**。当然，SIFT特征不仅仅是scale invariant，同样一张图片，如果改变以下一些特征仍能取得很好的结果：
* 尺度(Scale)
* 旋转(Rotation)
* 亮度(Illumination)
* 视角(Viewpoint)
例如，假设我们需要在下图中识别出图1所示的物体(object)，尽管图1中物体的视角和尺度和目标图中对应物体的不同，使用SIFT特征仍能准确的定位出该物体

![image](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/search_img.png) 
![image](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/object.png)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图一




