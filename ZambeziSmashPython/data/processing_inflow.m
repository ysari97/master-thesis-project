clear all
close all
clc

load zambezi_inflows_nov74_dec80.txt

%% Itezhitezhi

qInfITT = zambezi_inflows_nov74_dec80(:,1) ;
qKafueFlats = zambezi_inflows_nov74_dec80(:,2);
qInfKa = zambezi_inflows_nov74_dec80(:,3);
qInfCb = zambezi_inflows_nov74_dec80(:,4);
qShire = zambezi_inflows_nov74_dec80(:,5);

q = qInfITT(3:end); % Starting from January 1975

for i = 1 : 12 : length(q)-11
    q_split(:,i) = q(i:i+11) ;
end
q_split = q_split(:,1:12:end);

day_month = [31 28 31 30 31 30 31 31 30 31 30 31];

inflow_d = size(q_split);
for j = 1 : size(q_split,2)
    for i = 1:size(q_split,1)
        inflow_d(i,j) = q_split(i,j)./(day_month(i)*24*3600);
    end
end

qInfITT_d = inflow_d(:);
qInfITT_d = [qInfITT(1)./(30*3600*24) ; qInfITT(2)./(31*3600*24); qInfITT_d];

clear q q_split inflow_d

%% Kafue Flats

q = qKafueFlats(3:end); % Starting from January 1975

for i = 1 : 12 : length(q)-11
    q_split(:,i) = q(i:i+11) ;
end
q_split = q_split(:,1:12:end);

day_month = [31 28 31 30 31 30 31 31 30 31 30 31];

inflow_d = size(q_split);
for j = 1 : size(q_split,2)
    for i = 1:size(q_split,1)
        inflow_d(i,j) = q_split(i,j)./(day_month(i)*24*3600);
    end
end

qKafueFlats_d = inflow_d(:);
qKafueFlats_d = [qKafueFlats(1)./(30*3600*24) ; qKafueFlats(2)./(31*3600*24); qKafueFlats_d];

clear q q_split inflow_d

%% Kariba

q = qInfKa(3:end); % Starting from January 1975

for i = 1 : 12 : length(q)-11
    q_split(:,i) = q(i:i+11) ;
end
q_split = q_split(:,1:12:end);

day_month = [31 28 31 30 31 30 31 31 30 31 30 31];

inflow_d = size(q_split);
for j = 1 : size(q_split,2)
    for i = 1:size(q_split,1)
        inflow_d(i,j) = q_split(i,j)./(day_month(i)*24*3600);
    end
end

qInfKa_d = inflow_d(:);
qInfKa_d = [qInfKa(1)./(30*3600*24) ; qInfKa(2)./(31*3600*24); qInfKa_d];

clear q q_split inflow_d

%% Cahora Bassa

q = qInfCb(3:end); % Starting from January 1975

for i = 1 : 12 : length(q)-11
    q_split(:,i) = q(i:i+11) ;
end
q_split = q_split(:,1:12:end);

day_month = [31 28 31 30 31 30 31 31 30 31 30 31];

inflow_d = size(q_split);
for j = 1 : size(q_split,2)
    for i = 1:size(q_split,1)
        inflow_d(i,j) = q_split(i,j)./(day_month(i)*24*3600);
    end
end

qInfCb_d = inflow_d(:);
qInfCb_d = [qInfCb(1)./(30*3600*24) ; qInfCb(2)./(31*3600*24); qInfCb_d];

clear q q_split inflow_d

%% Shire

q = qShire(3:end); % Starting from January 1975

for i = 1 : 12 : length(q)-11
    q_split(:,i) = q(i:i+11) ;
end
q_split = q_split(:,1:12:end);

day_month = [31 28 31 30 31 30 31 31 30 31 30 31];

inflow_d = size(q_split);
for j = 1 : size(q_split,2)
    for i = 1:size(q_split,1)
        inflow_d(i,j) = q_split(i,j)./(day_month(i)*24*3600);
    end
end

qShire_d = inflow_d(:);
qShire_d = [qShire(1)./(30*3600*24) ; qShire(2)./(31*3600*24); qShire_d];

clear q q_split inflow_d

