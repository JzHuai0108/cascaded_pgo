# densify_poses

Depends on ceres solver which is wrapped in ceres_catkin.

# Build

## With ROS
```
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone --recursive git@bitbucket.org:JzHuai0108/densify_poses.git

wstool init
wstool merge densify_poses/dependencies.rosinstall
wstool update -j 8
cd ..
catkin build densify_poses -DCMAKE_BUILD_TYPE=Release -j4

```

