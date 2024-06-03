# cascaded_pgo
Given consecutive poses obtained by a drifting odometer,
and sparse key poses obtained by an optimization backend,
densify the sparse poses with the dense consecutive odometry poses,
by using a pose graph optimization on SE3.
The output poses will use the world frame of the sparse key poses.
The original world frame of the dense odometry poses does not affect the final trajectory.

# Build
This program depends on ceres solver.
## ceres solver
```
cd /home/$USER/Documents/slam_src
git clone --recursive https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
git checkout 2.1.0
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/home/$USER/Documents/slam_devel -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_EXAMPLES:BOOL=OFF -DBUILD_TESTING:BOOL=OFF \
    -DGLOG_INCLUDE_DIR_HINTS=/usr/include -DGLOG_LIBRARY_DIR_HINTS=/usr/lib

make install
```

## With ROS
```
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone --recursive git@bitbucket.org:JzHuai0108/cascaded_pgo.git

wstool init
wstool merge cascaded_pgo/dependencies.rosinstall
wstool update -j 8
cd ..
catkin build cascaded_pgo -DCMAKE_BUILD_TYPE=Release -j4 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCeres_DIR=/home/$USER/Documents/slam_devel/lib/cmake/Ceres \
  -DGLOG_INCLUDE_DIR_HINTS=/usr/include -DGLOG_LIBRARY_DIR_HINTS=/usr/lib

```

# Run
```
densify_exe=/media/jhuai/docker/cascaded_pgo_ws/devel/lib/cascaded_pgo/densifyPoses
cd $OUTPUTDIR
$densify_exe $OUTPUTDIR/trajectory.txt $OUTPUTDIR/keyframeTrajectory.txt
```

