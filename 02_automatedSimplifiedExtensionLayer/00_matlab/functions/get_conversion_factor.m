function con_fac = get_conversion_factor(unit_1,unit_2,name_flow)
%% unit_1 * con_fact = unit_2
con_fac = 0;

if strcmp(unit_2,'kg')&&strcmp(unit_1,'kg')
    con_fac = 1;
elseif strcmp(unit_2,'Nm3')&&strcmp(unit_1,'Nm3')
    con_fac = 1;
elseif strcmp(unit_2,'MJ')&& strcmp(unit_1,'MJ')
    con_fac = 1;
elseif strcmp(unit_2,'MJ')&& strcmp(unit_1,'kWh')
    con_fac = 1/3.6;
% elseif strcmp(name_flow,'METHANE')&& strcmp(unit_2,'kg')&& strcmp(unit_1,'m3')
%     con_fac = 0.7175;
end

%%  check if unit conversion is known
if con_fac == 0
    disp(['Can not convert ', unit_1 ,' to ', unit_2, '. Please contact Raoul Meys to resolve the problem.'] );
end
end