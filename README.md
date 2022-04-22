# densify_poses
Given consecutive poses obtained by a drifting odometer,
and sparse key poses obtained by an optimization backend,
densify the sparse poses with the dense consecutive odometry poses,
by using a pose graph optimization on SE3.
The output poses will use the world frame of the sparse key poses.
The original world frame of the dense odometry poses does not affect the final trajectory.

This program depends on ceres solver which is wrapped in ceres_catkin.

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

# Run
```
densify_exe=/media/jhuai/docker/densify_poses_ws/devel/lib/densify_poses/densifyPoses
cd $OUTPUTDIR
$densify_exe $OUTPUTDIR/trajectory.txt $OUTPUTDIR/keyframeTrajectory.txt
```

