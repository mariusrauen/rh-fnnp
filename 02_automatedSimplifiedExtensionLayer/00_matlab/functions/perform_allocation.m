function output = perform_allocation( Atoallocate, B, F, Utilities, Amatrixforallocationfactors, meta_data_processes, meta_data_flows)

%implementation of Eq. (6) and (7) in Jung et al. 2013
%Eq. (6): A_alloc = [U.*(A*T_1)]*C
%Eq. (7): B_alloc = [B*T_0]*C
%notation according to Jung et al. 2013
%coded by Björn Bahl, #E$T Inc. :P


%% Analyze of Matrices
A = Amatrixforallocationfactors;

%matrix size

[m,n] = size(A);

%output flows
IsOutFlow = Atoallocate>0;
%IsOutFlow = A>0;
NumOutFlowsOfProcess = sum(IsOutFlow,1);
IsMultiOutProcess = NumOutFlowsOfProcess>1;
IsSingleOrNonOutProcess = NumOutFlowsOfProcess<=1; % non-output is not discussed in Jung et al.
NumSingleOrNonOutProcess = sum(IsSingleOrNonOutProcess); 
NumOutFlowsOfMultiOutProcess = sum(sum(IsOutFlow(:,IsMultiOutProcess),1));
n_beta(IsSingleOrNonOutProcess) = 1; % vector to determine size of matrices
n_beta(IsMultiOutProcess) = 1 + NumOutFlowsOfProcess(IsMultiOutProcess); % T_1 requires 1+#OutFlows columns per multi-output process

%'only-output' Matrix, required for allocation
A_out = zeros(m,n); 
A_out(IsOutFlow) = A(IsOutFlow);
A_sumOut = sum(A_out);

%% Construct T_0,T_1 and U matrix, based on column generation

%Transformation matrices T_0, T_1 are based on (n x n) identity matrix
%T_1: Expand identity matrix to (n x beta) transformation matrix T_1, i.e. n_beta 
%T_0: like T_1, only zeros at expanded columns
%U: Function matrix U is of same size as A*T_1, i.e. (m x(n+beta))
Identity_n = eye(n);
T_0 = zeros(n,sum(n_beta));
T_1 = zeros(n,sum(n_beta));
U = zeros(m,sum(n_beta));

c=1;%column counter
for i = 1:n %for all processes
for j = 1:n_beta(i) %for all flows in each process
    T_1(:,c)=Identity_n(:,i);   
 	if(NumOutFlowsOfProcess(i)<=1) %is single-functional or non-output process
        T_0(:,c)=Identity_n(:,i);
        U(:,c)=1; 
    else %allocation
        if j==1 %original multi-product process
            T_0(:,c)=Identity_n(:,i);
            U(:,c)=1; %'... and ones elsewhere'
            U(IsOutFlow(:,i),c)=0; %'has zeros in the rows of the functional flows ...'             
        else %duplicated columns
            T_0(:,c)=zeros(n,1);
            IdxOutFlows=find(IsOutFlow(:,i),j-1); %identify position of flows
            U(IdxOutFlows(end),c)=1;
        end
    end    
c=c+1;
end
end

%% Construct C matrix, based on row generation
%   The general formulation to obtain C-factors is not described in Jung et al. 2013:
%   allocation is based on weighted output of single flow relative to all flows of on process
%   Errata in Jung et al. 2013: 
%   size of C = ((n+b)x m) is not correct! m only valid in example-case
%   correc: columns of C must be calculated according to equation below
C=zeros(sum(n_beta),NumSingleOrNonOutProcess+NumOutFlowsOfMultiOutProcess);

% expanded meta-data array
new_meta_data_processes=cell(size(meta_data_processes,1),NumOutFlowsOfMultiOutProcess+NumSingleOrNonOutProcess);
new_meta_data_processes(:,1)= meta_data_processes(:,1);

r=1; %row counter
c=1; %column counter
for i=1:n %for all processes
    
        
    if i==n
        pause=1;
    end
    
    if NumOutFlowsOfProcess(i)<=1 %is single-functional or non-output process 
        C(r,c)=1;%add row of identify matrix
        new_meta_data_processes(:,c+1) = meta_data_processes(:,i+1); %save original metadata column,+1 due to 'rownames' in first column of meta_data_processes
    else %is multi-functional process
        %add allocation factor row 
        IdxOutFlows = find(Atoallocate(:,i)>0); %store row number of output flows in A for current process
        for j=1:NumOutFlowsOfProcess(i) %for all output flows of multi-func process
            C(r,c+j-1)=A_out(IdxOutFlows(j),i)/A_sumOut(i); %HERE WE REALLY CALCULATE THE ALLOCATION FACTORS            
            new_meta_data_processes(:,(c+j-1)+1) = meta_data_processes(:,i+1); %duplicate original metadata column
            
            process_name = meta_data_processes{1,i+1};
            flow_name = meta_data_flows{IdxOutFlows(j)+1,1};
            if contains(flow_name,' BY ')
            flow_name = flow_name(1:(end-length(process_name)-3));
            end
            new_meta_data_processes(4,(c+j-1)+1) = cellstr(flow_name);
            
        end
        %add identity matrix for all outputs of this multi-output process
        C(r+1:r+NumOutFlowsOfProcess(i),c:c+NumOutFlowsOfProcess(i)-1)=eye(NumOutFlowsOfProcess(i)); 
        r=r+NumOutFlowsOfProcess(i);
        c=c+NumOutFlowsOfProcess(i)-1;
    end   
c=c+1;
r=r+1;
end

%% Calculate allocation

A_alloc = (U.*(Atoallocate*T_1))*C;
B_alloc = (B*T_0)*C;
F_alloc = (F*T_0)*C;
Utilities_alloc = (Utilities*T_0)*C;

%% override data in Model
output.A = A_alloc;
output.B = B_alloc;
output.F = F_alloc;
output.meta_data_processes = new_meta_data_processes;
output.Utilities = Utilities_alloc;
end