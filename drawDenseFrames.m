folder='densify_poses/test/data/';
ftxt = [folder, 'dataset-corridor1_512_16_poses.txt'];
kftxt = [folder, 'dataset-corridor1_512_16_kfposes.txt'];
dftxt = [folder, 'dense_poses.txt'];
datafiles = {ftxt, kftxt, dftxt};

fd = readmatrix(ftxt);
kfd = readmatrix(kftxt);
dfd = readmatrix(dftxt);
close all;
figure;
plot3(fd(:, 2), fd(:, 3), fd(:, 4), 'r');
hold on; axis equal; grid on;
plot3(kfd(:, 2), kfd(:, 3), kfd(:, 4), 'k');
plot3(dfd(:, 2), dfd(:, 3), dfd(:, 4), 'g');
legend('frames', 'keyframes', 'dense frames');

% figure;
% drawColumnsInMultipleFiles(datafiles, {'frames', 'keyframes', 'dense frames'}, ...
%    {'x', 'y', 'z'}, 2:4, 0);