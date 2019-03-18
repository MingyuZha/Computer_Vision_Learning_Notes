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

## The scale space
在现实世界中，物体往往只在某个特定的尺度（准确来说，是清晰度，或者分辨率）下才能被准确的识别出，例如你也许可以看见在桌子上有一块方糖，但是如果画面十分模糊，你是无法看到方糖的。
### Scale spaces
如果你需要识别的物体是一棵树，那么你就不需要识别出树上的叶子这些details。去除这些不必要的细节常用的手段就是是图片模糊化(blur)，为了在模糊的过程中不引入新的false details，一般使用**Gaussian Blur**。
创建一个scale space的方法是将原始输入图片不断的模糊化，下图就是一个例子：

![image](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/scalespace.jpg)

### Scale spaces in SIFT
在SIFT中，生成scale spaces除了对原始的输入图片做递进的模糊操作以外，我们还会将原始的输入图片做**Resize**操作，使其大小为原图的一半，然后对resize过后的图片再次进行模糊化操作，以此类推。


![image](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/sift-octaves.jpg)

由上图可见，同一列的图片是相同size的，它们共同组成了一个```octave```，上图共有4个octaves，每一个octave包含了5张图片，每张图片具有不同的```scale``` (the amount of blur)。

### The technical details
#### Octave and Scales
Octaves和scales的数量取决于原始图片的大小，一般需要由用户自己定义这个数量，但是，SIFT的创造者建议使用4个Octaves和5个blur levels。
#### The first octave
如果将原始的输入图片尺寸扩大一倍并且做抗锯齿操作（by blurring it），那么算法生成的keypoints将会是原来的**四倍以上**。Keypoints越多，算法的性能越好！
#### Blurring
从数学的角度来说，Blurring是指对图像中的像素做卷积操作，**Gaussian blur**的数学表达式为：

![equation](https://latex.codecogs.com/gif.latex?L%28x%2C%20y%2C%20%5Csigma%29%20%3D%20G%28x%2C%20y%2C%20%5Csigma%29%20*%20I%28x%2C%20y%29)

公式中的符号含义：
* L代表模糊后的图片
* G代表高斯滤波器
* I代表原始输入图片
* x, y是像素点坐标
* ![sigma](https://latex.codecogs.com/gif.latex?%5Csigma)
是**scale parameter**.可以把它当做是模糊的程度，值越大，越模糊
* *代表卷积操作

#### Amount of blurring
假设在某个特定的图片中模糊程度为![sigma](https://latex.codecogs.com/gif.latex?%5Csigma)，那么，下一张图的模糊程度就是k*![](https://latex.codecogs.com/gif.latex?%5Csigma)，k是由用户设定的常数











