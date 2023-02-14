close all
clear all

addpath('..\functions\')

pattern_period=400e-6;
pulse_period=20e-6;
ppp=60;
DC=0.3;
input_mode='ppp';
treset_on=1e-6;
tfall=100e-9;
trise=100e-9;
current_time=0;
current_vector=[];
reset_time=0;
reset_vector=[];
voltage=50e-3;

patterns=[1,1,1,0,0,0;...
          0,0,1,0,0,0;...
          0,0,1,0,0,1;...
          0,1,0,0,1,0;...
          1,0,0,0,1,0;...
          0,0,0,0,1,1;...
          0,1,0,1,0,0;...
          0,0,0,1,0,0;...
          1,0,0,1,0,1];

patterns=horzcat(patterns,[1 0 0 0 1 0 0 0 1]');
patterns=horzcat(patterns,[1 1 1 0 0 0 0 0 0]');
patterns=horzcat(patterns,[1 1 1 0 0 1 0 0 1]');

load('..\..\Python\matlab_matrix.mat')
patterns=noisy_patterns';

if strcmpi(input_mode,'ppp')
    pulse_period=pattern_period/ppp;
elseif strcmpi(input_mode,'pulse_period')
    ppp=pattern_period/pulse_period;
end

for k=1:size(patterns,1)
    for h=1:size(patterns,2)
        for i=1:ppp
            for j=1:4
                switch j
                    case 1
                        current_vector=vertcat(current_vector,[current_time,0]);
                        current_time=current_time+trise;
                    case 2
                        current_vector=vertcat(current_vector,[current_time,voltage*patterns(k,h)]);
                        current_time=current_time+(pulse_period*DC-trise);
                    case 3
                        current_vector=vertcat(current_vector,[current_time,voltage*patterns(k,h)]);
                        current_time=current_time+tfall;
                    case 4 
                        current_vector=vertcat(current_vector,[current_time,0]);
                        current_time=current_time+(pulse_period*(1-DC)-tfall);                
                end
            end
        end
        if k==size(patterns,1)
            for j=1:4
                switch j
                    case 1
                        reset_vector=vertcat(reset_vector,[current_time,0]);
                        reset_time=current_time+trise;
                    case 2
                        reset_vector=vertcat(reset_vector,[reset_time,voltage]);
                        reset_time=reset_time+(treset_on);
                    case 3
                        reset_vector=vertcat(reset_vector,[reset_time,voltage]);
                        reset_time=reset_time+tfall;
                    case 4 
                        reset_vector=vertcat(reset_vector,[reset_time,0]);
                end
            end
        end
        current_vector=vertcat(current_vector,[current_time,0]);
        current_time=current_time+(tfall+trise+treset_on);  
    end
    fileID = fopen(sprintf('..\\..\\voltage_source_%d.txt',k),'w');
    fprintf(fileID,'%.6g %.3g\n',current_vector');
    fclose(fileID);
    if k==size(patterns,1)
        fileID = fopen(sprintf('..\\..\\reset_signal.txt'),'w');
        fprintf(fileID,'%.6g %.3g\n',reset_vector');
        fclose(fileID);
        reset_vector=[];
        reset_time=0;
    end
    current_vector=[];
    current_time=0;
end


runMode='-b';

exe_file="C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe";
system_command = strcat('"',exe_file,'" -run',{' '},runMode,{' '},'..\SNN_TSM.asc');
[status,cmdout]=system(system_command,'-echo');

RAW_DATA=LTspice2Matlab('..\..\..\..\SNN_TSM.raw');

for i=1:size(RAW_DATA.variable_name_list,2)
    if strcmpi(RAW_DATA.variable_name_list{1,i},'v(reset_neuron)')
        reset_signal_pos=i;
        reset_signal=RAW_DATA.variable_mat(i,:);
    end
end

reset_f = find([0,reset_signal,0]>25e-3);
reset_p = reset_f(1:2:end-1);  % Start indices
reset_y = reset_f(2:2:end)-reset_p;  % Consecutive onesr counts

for i=1:size(RAW_DATA.variable_name_list,2)
    for j=1:6
        if strcmpi(RAW_DATA.variable_name_list{1,i},sprintf('v(output_%d)',j))
            eval(sprintf('output_signal_%d=RAW_DATA.variable_mat(%d,:);',j,i));
            eval(sprintf('output_signal_%d_pos=%d;',j,i));
        end
    end
end

for i=0:length(reset_p)
    if i==0
        section_idx=[1,reset_p(1)];
    elseif i<length(reset_p)
        section_idx=[reset_p(i) reset_p(i+1)];
    else
        section_idx=[reset_p(i) length(reset_signal)];
    end

    for j=1:6
        eval(sprintf('signal2proc=output_signal_%d(section_idx(1):section_idx(2));',j));

        signal_f = find(diff(signal2proc>0.7e-3));
        signal_p = signal_f(1:2:end-1);  % Start indices
        %signal_y = signal_f(2:2:end)-signal_p;  % Consecutive onesr counts

        outputs(j,i+1)=length(signal_p);
    end
end
