# Monocular-Visual-Odometry
Visual odometry is the process of determining the location and orientation (trajectory) of a camera by analyzing a sequence of images. Visual odometry is used in a variety of applications, such as mobile robots, self-driving cars, and unmanned aerial vehicles. Odometry in Robotics is a more general term, and often refers to estimating not only the distance traveled but the entire trajectory of a moving robot.
# The algorithm
1- Capture images: I(t), I (t+1)<br />
2- Undistort the above images (if needed)<br />
3- Use (FAST, SIFT, SUFT, ORB) algorithm to detect features in I(t), and track those features to I(t+1). A new detection is triggered if the number of features drops below a certain threshold. <br />
4- Use Nister’s 5-point algorithm with RANSAC to compute the essential matrix.<br />
5- Estimate R, t from the essential matrix that was computed in the previous step.<br />
6- Take scale information from some external source (like a speedometer), and concatenate the translation vectors and rotation matrices.<br />
### [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

## Results 
![alt text](https://github.com/abwerby/Monocular-Visual-Odometry/blob/master/map.png)
![alt text](https://github.com/abwerby/Monocular-Visual-Odometry/blob/master/Figure_1.png)  
