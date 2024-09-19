function [ Model,Amatrixforallocationfactors , possible ] = get_price_matrix_A(Model)

possible=1;

price_vector = cell2mat(Model.meta_data_flows(2:end,7));

if size(price_vector,1)<size(Model.meta_data_flows(2:end,7),1) || ~any(price_vector <= 0)
   possible=0;
   error('No price allocation possible, because not all prices are given or one price is zero.')
   return
end

% get prices for all outputs for all processes
A = Model.matrices.A.mean_values;
A(A<0) = 0;

price_matrix = repmat(price_vector,1,size(A,2));

Amatrixforallocationfactors = A.*price_matrix;

%% test if allocation according to price is needed and reset 'allocation types' to 2
% without utilities
lns_utilities = cell2mat(Model.meta_data_flows(2:end,2)) == 2;
A(lns_utilities,:) = 0;
% if there is only one output -> no allocation has to be performed -> we can set this process to 0
counter = A;
counter(counter>0) = 1;
counter = sum(counter,1);
lns_oneoutput = counter == 1;
A(:,lns_oneoutput) = 0;
A(A>0) = price_matrix(A>0);

Model.meta_data_processes(10,[false,lns_oneoutput]) = {0};

% get all processes that might be allocated according to price
allocationType = nan(size(Model.meta_data_processes(10,:)));
idx_strings = cellfun(@ischar,Model.meta_data_processes(10,:));
allocationType(~idx_strings) = cell2mat(Model.meta_data_processes(10,~idx_strings));
allocationType(idx_strings) = 0;
lns_pricealloc = allocationType(2:end) == 2;
A(:,~lns_pricealloc) = 0;

% find all relevant processes only:
nonzero_cols = ~(all(A==0,1));
A_nonzero_cols = A(:,nonzero_cols);

% get max, min, and factor between output prices
maxA = max(A_nonzero_cols);
minA = zeros(size(maxA));
for i=1:size(minA,2)
    A_min = A_nonzero_cols(:,i);
    minA(1,i) = min(A_min(A_min~=0));
end
factor = maxA./minA;

if any(factor <= 0 | isnan(factor))
    % Display an error message
    error('Some price allocation factors are non-positive or NaN.');
end

% create the full factor vector and extract only those where allocation
% factor is > 5
fullFactor = zeros(1,size(A,2));
fullFactor(nonzero_cols) = factor;
priceAllocation = fullFactor>5;

%% set values to price allocation
Model.meta_data_processes(10,[false,priceAllocation]) = {3};


end