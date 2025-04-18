%% Seismic Signal Segmentation

%% Step 1: Load Data
% load seis_de
dt = 0.002;
t = (0:length(seis_de)-1) * dt;

%% Step 2: Parameters
window_size = 2.048;
start_time = 2260;
end_time = 2290;
segment_duration = 1.024;
output_folder = "C:\Users\Administrator\Desktop\data\field data\site_1_segment";

%% Step 3: Analyze and Plot
[seis_de_mov, t_mov, moving_abs_signal, t_movmean, peak_times] = ...
    movingmean_and_peaks(seis_de, t, dt, start_time, end_time, window_size);

%% Step 4: Manual Picking and Save Segments
dc = [];
i = 0;
button = 1;
disp('Left click to pick, right click to finish.');
while button ~= 3
    i = i + 1;
    [x, ~, button] = ginput(1);
    dc(i, 2) = x;
    dc(i, 1) = 0;
    fprintf('Picked time %d: %.3f s\n', i, x);
end
save('dc.mat', 'dc');

% Save segments on the selected window only
num_segments = size(dc, 1);
for i = 1:num_segments
    center_time = dc(i, 2);
    start_seg = max(center_time - segment_duration, t_mov(1));
    end_seg = min(center_time + segment_duration, t_mov(end));
    start_idx = find(t_mov >= start_seg, 1);
    end_idx = find(t_mov <= end_seg, 1, 'last');

    segment_time = t_mov(start_idx:end_idx);
    segment_data = seis_de_mov(start_idx:end_idx);

    filename = sprintf('segment_%.3f.txt', center_time);
    filepath = fullfile(output_folder, filename);
    data_to_save = [segment_time(:), segment_data(:)];
    writematrix(data_to_save, filepath, 'Delimiter', '\t');
    fprintf('Segment %d saved to %s\n', i, filepath);
end

disp('All segments have been picked and saved successfully.');

% Plot picked segments on selected window (raw and moving average)
figure;

% Raw data
ax1 = subplot(2,1,1);
set(ax1, 'Position', [0.1 0.54 0.85 0.4]);
plot(t_mov, seis_de_mov, 'k', 'LineWidth', 1.5); hold on;
for i = 1:num_segments
    center_time = dc(i, 2);
    center_idx = find(t_mov >= center_time, 1);
    plot(t_mov(center_idx), 0, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);

    start_time = center_time - segment_duration;
    end_time = center_time + segment_duration;
    xline(start_time, '--g', sprintf('Start %d', i), 'LabelVerticalAlignment', 'middle', 'LineWidth', 1.5);
    xline(end_time, '--b', sprintf('End %d', i), 'LabelVerticalAlignment', 'middle', 'LineWidth', 1.5);
end
ylabel('Amplitude', 'FontSize', 24);
set(gca, 'FontName', 'Arial', 'FontSize', 24);
set(ax1, 'XTickLabel', [], 'Xtick', []);
hold off;

% Moving average
ax2 = subplot(2,1,2);
set(ax2, 'Position', [0.1 0.12 0.85 0.4]);
plot(t_movmean, moving_abs_signal, 'r', 'LineWidth', 1.5); hold on;
for i = 1:num_segments
    center_time = dc(i, 2);
    center_idx = find(t_movmean >= center_time, 1);
    plot(t_movmean(center_idx), moving_abs_signal(center_idx), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
end
xlabel('Time (s)', 'FontSize', 24);
ylabel('Averaged Amplitude', 'FontSize', 24);
set(gca, 'FontName', 'Arial', 'FontSize', 24);
hold off;