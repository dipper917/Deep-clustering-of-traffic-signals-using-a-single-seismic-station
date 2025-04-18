function [seis_de_mov, t_mov, moving_abs_signal, t_movmean, peak_times] = movingmean_and_peaks(seis, t, dt, start_time, end_time, window_size)
% window extraction and peak detection
% Inputs:
%   seis         - seismic signal vector
%   t            - corresponding time vector
%   dt           - sampling interval
%   start_time   - start of analysis window
%   end_time     - end of analysis window
%   window_size  - size of the smoothing window (in seconds)
% Outputs:
%   seis_de_mov       - seismic signal within the selected window
%   t_mov             - time vector for seis_de_mov
%   moving_abs_signal - smoothed absolute amplitude signal
%   t_movmean         - time vector for moving average signal
%   peak_times        - detected peak times in the smoothed signal

    % Extract time window
    time_idx = find(t >= start_time & t <= end_time);
    seis_de_mov = seis(time_idx);
    t_mov = t(time_idx);

    % Compute moving average
    window_samples = round(window_size / dt);
    moving_abs_signal = movmean(abs(seis_de_mov), window_samples);
    t_movmean = t_mov;

    % Detect peaks
    [pks, locs] = findpeaks(moving_abs_signal, 'MinPeakDistance', round(1/dt), 'MinPeakHeight', 0);
    peak_times = t_movmean(locs);

    figure;
    subplot(2, 1, 1);
    plot(t_mov, seis_de_mov, 'k');
    xlabel('Time (sec)', 'FontSize', 14);
    ylabel('Amplitude', 'FontSize', 14);
    title(sprintf('Raw Seismic Signal (%d-%ds interval)', start_time, end_time), 'FontSize', 16);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 18);

    subplot(2, 1, 2);
    plot(t_movmean, moving_abs_signal, 'r'); hold on;
    plot(peak_times, moving_abs_signal(locs), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
    xlabel('Time (s)', 'FontSize', 14);
    ylabel('Amplitude', 'FontSize', 14);
    title(sprintf('Sliding Window Mean (%.3fs)', window_size), 'FontSize', 16);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 18);
    hold off;
end