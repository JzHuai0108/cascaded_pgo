import selected_bags as sb
import argparse
import numpy as np
import os
import scipy.spatial.transform
import shutil
import subprocess

def T_from_pq(pq):
    T = np.identity(4)
    T[0:3, 0:3] = scipy.spatial.transform.Rotation.from_quat(pq[3:]).as_matrix()
    T[0:3, 3] = pq[0:3]
    return T

def pq_from_T(T):
    pq = np.zeros(7)
    pq[0:3] = T[0:3, 3]
    pq[3:] = scipy.spatial.transform.Rotation.from_matrix(T[0:3, 0:3]).as_quat()
    return pq

def transform_poses(bagname, resultdir, E_T_tls):
    tls_loc_file = os.path.join(resultdir, "ref_trajs_all", bagname, "tls_T_xt32.txt")
    val_strs = []
    with open(tls_loc_file, "r") as f:
        for line in f:
            val_strs.append(line.strip().split())
    outputdir = os.path.join(resultdir, "pgo", bagname)
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    outputfile = os.path.join(resultdir, "pgo", bagname, "utm50r_T_xt32.txt")
    outputkittiposes = os.path.join(resultdir, "pgo", bagname, "utm50r_T_xt32_kitti.txt")
    outputkittitimes = os.path.join(resultdir, "pgo", bagname, "times_kitti.txt")
    kp = open(outputkittiposes, "w")
    kt = open(outputkittitimes, "w")
    with open(outputfile, "w") as f:
        for vstr in val_strs:
            pq = np.array([float(val) for val in vstr[1:]])
            W_T_L = T_from_pq(pq)
            E_T_L = E_T_tls @ W_T_L
            E_pq_L = pq_from_T(E_T_L)
            f.write(f'{vstr[0]}')
            for val in E_pq_L:
                f.write(f' {val:.09f}')
            f.write("\n")

            kt.write(f"{vstr[0]}\n")
            kp.write(f"{E_T_L[0, 0]:.09f} {E_T_L[0, 1]:.09f} {E_T_L[0, 2]:.09f} {E_T_L[0, 3]:.09f} ")
            kp.write(f"{E_T_L[1, 0]:.09f} {E_T_L[1, 1]:.09f} {E_T_L[1, 2]:.09f} {E_T_L[1, 3]:.09f} ")
            kp.write(f"{E_T_L[2, 0]:.09f} {E_T_L[2, 1]:.09f} {E_T_L[2, 2]:.09f} {E_T_L[2, 3]:.09f}\n")

    kp.close()
    kt.close()


def seq_available(bagname, datadir, resultdir):
    run = bagname.split("/")[1]
    odom_file = os.path.join(resultdir, "kissicp", bagname, f"{run}_aligned_poses_tum.txt")
    tls_loc_file = os.path.join(resultdir, "ref_trajs_all", bagname, "tls_T_xt32.txt")
    gnss_loc_file = os.path.join(resultdir, "ref_trajs_all", bagname, "utm50r_T_x36dimu.txt")
    imu_file = os.path.join(datadir, bagname, "x36d", "imu.txt")
    if not os.path.isfile(odom_file):
        print(f"Missing odometry {odom_file}")
        return False
    if not os.path.isfile(tls_loc_file):
        print(f"Missing ref TLS traj {tls_loc_file}")
        return False
    if not os.path.isfile(gnss_loc_file):
        print(f"Missing GNSS/INS traj {gnss_loc_file}")
        return False
    if not os.path.isfile(imu_file):
        print(f"Missing IMU file {imu_file}")
        return False
    return True

def run_pgo(datadir, resultdir, wsdir, shell_script, logfile):
    # selected_bags = sb.selected_bags2
    selected_bags = sb.selected_bags

    E_pq_tls = sb.E_pq_tls.split()
    E_T_tls = T_from_pq(E_pq_tls)

    for bagname in selected_bags.keys():
        vals = selected_bags[bagname]
        if vals[0]:
            print("Transforming poses for", bagname)
            transform_poses(bagname, resultdir, E_T_tls)
            continue
        close_z_pairs = vals[1]
        print("Running PGO for ", bagname)
        if not seq_available(bagname, datadir, resultdir):
            continue
        result = subprocess.run([shell_script, bagname, datadir, resultdir, wsdir, close_z_pairs], 
                                capture_output=True, text=True)
        with open(logfile, "a") as f:
            f.write(f"Processing {bagname}\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("Error:\n")
                f.write(result.stderr)
            f.write("\n")

        print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)

def copy_pgo_trajs(resultdir, outbasename):
    indir = os.path.join(resultdir, "pgo")
    outdir = os.path.join(resultdir, outbasename)
    count = 0
    copyfilenames = ["utm50r_T_xt32.txt", "utm50r_T_xt32_kitti.txt", "times_kitti.txt"]
    for bagname in sb.selected_bags.keys():
        bagoutdir = os.path.join(outdir, bagname)
        if not os.path.isdir(bagoutdir):
            os.makedirs(bagoutdir)
        for filename in copyfilenames:
            infile = os.path.join(indir, bagname, filename)
            outfile = os.path.join(bagoutdir, filename)
            if not os.path.isfile(infile):
                print(f"Warn: Missing {infile}")
                continue
            else:
                shutil.copy(infile, outfile)
                count += 1
    print(f"Copied {count} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Run PGO on selected bags""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("datadir", help="Directory containing the sequence data, for getting the x36d IMU data.")
    parser.add_argument("resultdir", help="Directory to store the results, for getting kissicp odometry, lio loc results, and gnss data")
    parser.add_argument("wsdir", help="Directory of the cascaded_pgo workspace")

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    shell_script = os.path.join(script_dir, "pgo_core.sh")

    logfile = os.path.join(args.resultdir, "pgo", "run_pgo.log")
    run_pgo(args.datadir, args.resultdir, args.wsdir, shell_script, logfile)

    # copy_pgo_trajs(args.resultdir, 'full_trajs')
    print("Done")
