#!/bin/bash
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 dataname datadir resultdir wsdir close_z_pairs"
    exit 1
fi

dataname=$1
datadir=$2
resultdir=$3
wsdir=$4
close_z_pairs=$5


cd $wsdir
source devel/setup.bash

echo "Running PGO on $dataname"
date=${dataname%%/*} # the first part of $bag
run=${dataname#*/} # the second part of $bag
odom_file=$resultdir/kissicp/$dataname/"$run"_aligned_poses_tum.txt
tls_loc_file=$resultdir/fwd_bwd_loc/$dataname/ref_tls_T_xt32.csv
gnss_loc_file=$resultdir/fwd_bwd_loc/$dataname/utm50r_T_x36dimu.txt
imu_file=$datadir/$dataname/x36d/imu.txt
output_path=$resultdir/pgo/$dataname
mkdir -p $output_path
E_T_tls_0921_5="245821.138870949 3380553.013607463 20.123183885 0.000857707 0.000124708 0.112339254 0.993669533"
rosrun cascaded_pgo pgo $odom_file $tls_loc_file $gnss_loc_file $imu_file $output_path --gnss_sigma_xy=0.1 --gnss_sigma_z=1.0 \
    --relative_trans_hori_sigma=0.5 --relative_trans_vert_sigma=0.3 --relative_rot_sigma=0.1 \
    --E_T_tls="$E_T_tls_0921_5" --gnss_sigma_z_bad=10.0 --imu_step=1 \
    --close_z_pairs="$close_z_pairs" --close_z_sigma=0.3

# for seq 20240116/data2, we set --close_z_sigma=0.5 --relative_trans_vert_sigma=0.1
# for other seqs, we set --close_z_sigma=0.3 --relative_trans_vert_sigma=0.3