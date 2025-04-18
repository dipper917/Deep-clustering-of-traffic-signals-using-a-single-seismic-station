%%%%%% fcnt to seg2 %%%%%%
%%% sort files %%%
clear all;
clc;
filelocation = "C:\Users\Administrator\Desktop\data\field data";
files = dir("C:\Users\Administrator\Desktop\data\field data\site_1.fcnt"); % input

files_num = length(files);
Files = cell(files_num,1);
for i = 1:files_num
    Files{i} = files(i).name;
end
[cs,index] = sort_nat(Files); 

%%% parameter setting %%%
ntrace = 1;  % number of traces per gather
vs_num = files_num - ntrace + 1;
dt = 0.002;  % sampling interval
de = 1;  % decimation rate

%%% read fcnt and write seg2 %%%
aimfilelocation = "C:\Users\Administrator\Desktop\data\field data\pas\"; % output directory
trace = cell(ntrace,1);
for i = 1:vs_num
    shot_num = i;
    seis_Z = [];
    for j = shot_num:shot_num + ntrace - 1
        trace{j-shot_num+1} = files(index(j));
        traces = rd_fcntPRO((strcat('C:\Users\Administrator\Desktop\data\field data\',trace{j-shot_num+1}.name)), 'b', 0, 0, 1.75, 'Z');

        % read Z component and apply decimation
        seis_Z = traces.data;
        seis_Z_decut = decimate(seis_Z, de); 
        seis_de(:, j-shot_num+1) = seis_Z_decut;
    end

    writeseg2(seis_de, dt*de, aimfilelocation + num2str(shot_num) + '_' + num2str(shot_num + ntrace - 1) + '_' + num2str(de) + 'decimate' + '_Z.sg2');
    display(num2str(shot_num) + "/" + num2str(vs_num) + " accomplished");
end


% N = length(seis_de);
[N, num_traces] = size(seis_de); % get number of samples and number of traces
t = (0:N-1) * dt;

plot(t, seis_de, 'black');
xlim([0, max(t)]);
xlabel('Time (sec)', 'fontsize', 20);
ylabel('Amplitude', 'fontsize', 20);
title('Station 1');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 18);