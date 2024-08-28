cd /home/jhuai/Documents/swift_vio_ws_rel
source devel/setup.bash

process_bag() {
    dataname=$1
    date=${dataname%%/*} # the first part of $bag
    run=${dataname#*/} # the second part of $bag
    odom_file=/media/jhuai/MyBookDuo1/jhuai/results/kissicp/$dataname/"$run"_aligned_poses_tum.txt
    tls_loc_file=/media/jhuai/MyBookDuo1/jhuai/results/ref_trajs_all/$dataname/tls_T_xt32.txt
    gnss_loc_file=/media/jhuai/MyBookDuo1/jhuai/results/ref_trajs_all/$dataname/utm50r_T_x36dimu.txt
    imu_file=/media/jhuai/MyBookDuo1/jhuai/data/zip/tmp/$dataname/x36d/imu.txt
    output_path=/media/jhuai/MyBookDuo1/jhuai/results/pgo/$dataname
    mkdir -p $output_path
    E_T_tls_0921_5="245821.138870949 3380553.013607463 20.123183885 0.000857707 0.000124708 0.112339254 0.993669533"
    bad_gnss_segments="1706000435.789999962 1706001326.019999981“ # ”1706001026.019999981"
    close_z_pairs="1706000971.339999914 1706001307.220000029"
    rosrun cascaded_pgo pgo $odom_file $tls_loc_file $gnss_loc_file $imu_file $output_path --gnss_sigma_xy=0.1 --gnss_sigma_z=1.0 \
      --relative_trans_hori_sigma=0.5 --relative_trans_vert_sigma=0.3 --relative_rot_sigma=0.1 \
      --E_T_tls="$E_T_tls_0921_5" --gnss_sigma_z_bad=10.0 --imu_step=1 \
      --close_z_pairs="$close_z_pairs" --close_z_sigma=0.3
      # --bad_gnss_segments="$bad_gnss_segments" 
}

bagnames=(
20240123/data2
# 20240123/data3

# 20230921/data5
# 20230921/data3
# 20230920/data2
)

for bag in ${bagnames[@]}; do
    echo "Processing $bag"
    date=${bag%%/*} # the first part of $bag
    run=${bag#*/} # the second part of $bag
    process_bag $bag
done

