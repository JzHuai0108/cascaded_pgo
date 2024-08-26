cd /home/jhuai/Documents/swift_vio_ws_rel
source devel/setup.bash

process_bag() {
    dataname=$1
    date=${dataname%%/*} # the first part of $bag
    run=${dataname#*/} # the second part of $bag
    odom_file=/media/jhuai/MyBookDuo1/jhuai/results/kissicp/$dataname/"$run"_aligned_poses_tum.txt
    # odom_file=/media/jhuai/MyBookDuo1/jhuai/results/front_back_snapshots/$dataname/front/forward_states.txt
    tls_loc_file=/media/jhuai/MyBookDuo1/jhuai/results/ref_trajs_all/$dataname/tls_T_xt32.txt
    gnss_loc_file=/media/jhuai/MyBookDuo1/jhuai/results/ref_trajs_all/$dataname/utm50r_T_x36dimu.txt
    output_path=/media/jhuai/MyBookDuo1/jhuai/results/pgo/$dataname
    mkdir -p $output_path
    E_T_tls_0921_5="245821.138870949 3380553.013607463 20.123183885 0.000857707 0.000124708 0.112339254 0.993669533"
    cmd="rosrun cascaded_pgo pgo $odom_file $tls_loc_file $gnss_loc_file $output_path --gnss_sigma_xy=0.2 --gnss_sigma_z=1.0 \
      --relative_trans_sigma=2.0 --relative_rot_sigma=0.1 --use_nhc=false --nh_sigma=0.2 --huber_width=2.0" # --E_T_tls="$E_T_tls_0921_5"
    echo $cmd
    $cmd
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

