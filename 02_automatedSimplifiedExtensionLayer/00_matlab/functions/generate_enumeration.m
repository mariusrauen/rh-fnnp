function [string] = generate_enumeration(EnumerateString)

%% size of EnumerateString = 1 --> string = EnumerateString
if length(EnumerateString) == 1
    string = EnumerateString;
    return
end

if length(EnumerateString) == 2
    string = append(EnumerateString(1)," and ",EnumerateString(2));
    return
end

%% Else make XXX,XXX, and XX structure
string = EnumerateString(1);

for i = 2:length(EnumerateString)
    
    if i<length(EnumerateString)
    
        string = append(string,", ",EnumerateString(i));
        
    else
        
        string = append(string,", and ",EnumerateString(i));
        
    end
    
end

end