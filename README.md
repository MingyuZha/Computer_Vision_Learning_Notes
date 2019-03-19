
# Computer Vision Learning Notes

## 目录
* [Computer Vision Learning Notes](#computer-vision-learning-notes)
      * [SIFT](#sift)
         * [Why SIFT?](#why-sift)
      * [The scale space](#the-scale-space)
         * [Scale spaces](#scale-spaces)
         * [Scale spaces in SIFT](#scale-spaces-in-sift)
         * [The technical details](#the-technical-details)
      * [LoG approximations](#log-approximations)
         * [Laplacian of Gaussian](#laplacian-of-gaussian)
         * [The Con](#the-con)
         * [The Benefits](#the-benefits)
      * [Finding Keypoints](#finding-keypoints)
         * [Locate maxma/minima in DoG images](#locate-maxmaminima-in-dog-images)
         * [Find subpixel maxima/minima](#find-subpixel-maximaminima)
         
      
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
1. **Octave and Scales**
Octaves和scales的数量取决于原始图片的大小，一般需要由用户自己定义这个数量，但是，SIFT的创造者建议使用4个Octaves和5个blur levels。
2. **The first octave**
如果将原始的输入图片尺寸扩大一倍并且做抗锯齿操作（by blurring it），那么算法生成的keypoints将会是原来的**四倍以上**。Keypoints越多，算法的性能越好！
3. **Blurring**
从数学的角度来说，Blurring是指对图像中的像素做卷积操作，**Gaussian blur**的数学表达式为：

&emsp;&emsp;&emsp;&emsp;![equation](https://latex.codecogs.com/gif.latex?L%28x%2C%20y%2C%20%5Csigma%29%20%3D%20G%28x%2C%20y%2C%20%5Csigma%29%20*%20I%28x%2C%20y%29)

&emsp;&emsp;&emsp;&emsp;公式中的符号含义：

&emsp;&emsp;&emsp;&emsp;* L代表模糊后的图片

&emsp;&emsp;&emsp;&emsp;* G代表高斯滤波器

&emsp;&emsp;&emsp;&emsp;* I代表原始输入图片

&emsp;&emsp;&emsp;&emsp;* x, y是像素点坐标

&emsp;&emsp;&emsp;&emsp;* ![sigma](https://latex.codecogs.com/gif.latex?%5Csigma)是**scale parameter**.可以把它当做是模糊的程度，值越大，越模糊

&emsp;&emsp;&emsp;&emsp;* *代表卷积操作

4. **Amount of blurring**
假设在某个特定的图片中模糊程度为![](https://latex.codecogs.com/gif.latex?%5Csigma)，那么，下一张图的模糊程度就是k*![sigma](https://latex.codecogs.com/gif.latex?%5Csigma)，k是由用户设定的常数


## LoG approximations
在这一部分，我们将使用前面生成的模糊后的图片去生成另一组图片：```the Difference of Gaussians```**(DoG)**.这些DoG图片对发现图像中有益的**key points**十分有帮助。

### Laplacian of Gaussian
```The Laplacian of Gaussian``` operation的操作步骤是：先将原始图片通过高斯滤波器模糊化，然后计算图像的**second order derivatives**(也称作**laplacian**)。这步操作可以帮助定位到图中的边(edges)和角(corners)，这些边和角可以帮助发现图中的关键点信息。
但是，二阶导对于噪声十分敏感，因此需要在求二阶导之前对图像做降噪（模糊）处理，从而使得二阶导的求解更加的稳定。
问题在于，计算图像中所有像素位置的二阶导十分的computationally expensive，因此我们需要一条捷径。

### The Con
为了快速的生成LoG图像，我们使用到了**scale space**。我们计算两个连续scales之间的差值，或者说，the Difference of Gaussians，如图所示：

![image](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/sift-dog-idea.jpg)

这些DoG图片可近似等于**the Laplacian of Gaussian**，我们通过这样的方式将一个计算量十分庞大的过程化简成了一个简单的相减操作，大大提升了效率。

### The Benefits
仅仅是Laplacian of Gaussian图像还不够，因为它们并不是**scale invariant**的，也就是说它们取决于你对图像的模糊化程度，即![sigma](https://latex.codecogs.com/gif.latex?%5Csigma)的值

## Finding Keypoints
找到关键点分为两个步骤：
1. Locate maxima/minima in DoG images
2. Find subpixel maxima/minima

### Locate maxma/minima in DoG images
第一步是粗略的定位极大值和极小值，这一步很简单，只需要遍历图中的每个像素点，并将当前遍历的像素点与其邻接区域的像素点比较

![image](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/sift-maxima-idea.jpg)
**X**表示当前像素点，绿色的圈表示邻接点像素点，这样的话，对当前像素点，一共需要进行26次比较。
> 如果X位置像素点的值比其余26个像素点的值都大，或者都小，那么X就会被标记为"key point"

需要注意的是**我们不会再lowermost和topmost scales中检测keypoints**，因为我们没有足够的邻接点来进行检测工作，所以直接跳过这些scales上的像素点即可。
这一步完成之后，我们找到了”近似的“极大值点和极小值点，说它们是近似的，是因为真正的极大/小值点几乎不会落在某个具体的像素点上，它往往是落在像素点之间，但是我们没有办法获取像素点之间的数据。因此，我们必须通过数学计算的方法来定位这些**subpixel location**。

![image](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/sift-maxima-subpixel.jpg)

图中绿色的叉才是真正的极值点。

### Find subpixel maxima/minima
使用已有的像素值，subpixel的值可以被生成，方法是使用**the Taylor expansion of the image around the approximate key point**

> ![Taylor](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/Taylor.gif)

我们可以通过上面的泰勒展开式轻松的求解出该式的极值点(differentiate and eqaute to zero)

## Getting rid of low contrast keypoints
通过上一步可以生成许多的关键点，这些关键点中有一些会落在edge上，或者说它们并不具备足够的对比度(contrast)。在这种情况下，它们就不适合作为特征(features)。
### 去除low contrast特征
去除操作很简单，我们事先设定好一个阈值，如果在DoG图像中，当前像素点的值小于阈值，就去除。
### 去除edges
方案是计算在keypoint处的两个**相互垂直**方向上的梯度(gradient)，基于keypoint点周围的图像，存在三种不同的情况：
* 平摊区域(A flat region)：两个计算出来的梯度都很小
* 边(An edge)：其中一个梯度比较大（垂直于边的方向），另一个梯度比较小（沿着边的方向）
* 角落(A corner)：两个梯度都很大

**Corners**是很好的keypoints。所以我们只想要corners。如果两个gradients都足够大，我们就标记该候选点为key point，否则的话，拒绝。
数学上，我们通常使用```Hessian Matrix```来实现这一目的。

## Keypoint orientations
在获取了质量较高的keypoint之后，我们需要赋予每个keypoint方向(orientations)，方向的赋予使得keypoint具备**rotation invariant**特性。
### 如何收集梯度方向

![image](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/sift-orientation-window.jpg)

梯度值和方向可以通过如下公式来计算得到：

![image](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/sift-orientation-eqns.jpg)

在计算得到关键点周围所有像素点的梯度方向和大小后，生成直方图。由于梯度方向有360度，我们可以分成36个bin，每个bin代表10度。

![image](https://github.com/MingyuZha/Computer_Vision_Learning_Notes/raw/master/images/sift-orientation-histogram.jpg)

上图所示关键点的方向就是3（20-29度）
> Note: 扫描窗口的大小等于高斯滤波器的窗口大小

























