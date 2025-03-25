# cascaded_pgo
Given consecutive poses obtained by a drifting odometer,
and sparse key poses obtained by an optimization backend,
densify the sparse poses with the dense consecutive odometry poses,
by using a pose graph optimization on SE3.
The output poses will use the world frame of the sparse key poses.
The original world frame of the dense odometry poses does not affect the final trajectory.

# Build
This program depends on ceres solver.

## With ROS
```
mkdir -p catkin_ws/src
cd catkin_ws/src

git clone git@github.com:catkin/catkin_simple.git
git clone git@github.com:JzHuai0108/eigen_catkin.git
git clone git@bitbucket.org:JzHuai0108/ceres_catkin.git
git clone --recursive git@github.com:JzHuai0108/cascaded_pgo.git cascaded_pgo

cd ..
catkin build cascaded_pgo -DCMAKE_BUILD_TYPE=Release
# for ceres catkin, you may need to pass
# -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
# to catkin build per your system.
```

# Run
```
pgo_exe=/media/jhuai/docker/cascaded_pgo_ws/devel/lib/cascaded_pgo/pgo
cd $OUTPUTDIR
$pgo_exe $OUTPUTDIR/trajectory.txt $OUTPUTDIR/keyframeTrajectory.txt
```

