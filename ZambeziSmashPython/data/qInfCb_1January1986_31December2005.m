clear all
close all
clc

%% Modify Luangwa lateral inflow to Cahora Bassa in order to agree with the ZRA Report "Water Quality Analysis Volume 2 - Annex 1"
% Yearly average of Luangwa lateral contribution measured at Great Road Bridge = 557 m3/s

% Reduce the peaks above 3000 m3/s (NO registered flow above 3000 m3/s in the ZRA Report from 1960 to 1996)
load qInfCb_1January1986_31Dec2005.txt

qinfcb = nan(length(qInfCb_1January1986_31Dec2005),1) ;
for i=1:length(qInfCb_1January1986_31Dec2005)
    if qInfCb_1January1986_31Dec2005(i) > 3000
        qinfcb(i) = qInfCb_1January1986_31Dec2005(i).*0.70 ;
    else
        qinfcb(i) = qInfCb_1January1986_31Dec2005(i) ;
    end
end

for i=1:length(qinfcb)
    if qinfcb(i) > 3000
        qinfcb2(i) = qinfcb(i).*0.60 ;
    else
        qinfcb2(i) = qinfcb(i) ;
    end
end

cb_inflow = qinfcb2.*0.90 ;
cb_inflow(38:39) = qinfcb2(38:39) ;
cb_inflow(85:88) = qinfcb2(85:88) ;

cb_res = reshape(qInfCb_1January1986_31Dec2005,12,20) ; % 12 months in rows and 20 years in columns - avoi column 5,8,9 because of too high inflows
cb_avyear = mean([cb_res(:,1:4) , cb_res(:,6:7) , cb_res(:,10:end)],2) ; % average cyclostationary year (12 average monthly values)
cb_inflow(95:105) = [cb_avyear(11:12) ; cb_avyear(1:9)] ; % substitute outlier year November 1993 - September 1994
mean(cb_inflow) % 557 m3/s according to the report ; 579 m3/s according to this computations

figure
plot(qInfCb_1January1986_31Dec2005,'r','LineWidth',1.5)
hold on
plot(cb_inflow,'g','LineWidth',1.5)