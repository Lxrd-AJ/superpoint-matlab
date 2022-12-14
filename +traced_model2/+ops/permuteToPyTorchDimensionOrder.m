function varargout = permuteToPyTorchDimensionOrder(varargin)
% Function to permute the inputs to a placeholder function into 
% Python dimension ordering. The outputs will be unlabelled, and of the
% same data type as their corresponding inputs.

%   Copyright 2022 The MathWorks, Inc.

varargout = cell(1, nargin);
for i=1:nargin
    X = varargin{i};
    if isstruct(X) && isdlarray(X.value)
        % Input is a struct with 'value' and 'rank' fields.            
        X.value = permuteDlarrayToPythonDimensionOrder(X.value, X.rank);
        varargout{i} = X;
    else
        % Return the value unchanged; it is already in Python dimension
        % order.
        varargout{i} = X;
    end
end

end

function dlX = permuteDlarrayToPythonDimensionOrder(dlX, rank)
    if ~isempty(dims(dlX)) && ~any(dims(dlX) == 'U')
        % Input has SCBT dimension labels
        labels = dims(dlX);    
        switch labels
            case 'SSCB'
                permDLTToPython = [4 3 1 2]; % HWCN -> NCHW
            case 'SSC'
                permDLTToPython = [3 1 2]; % HWC -> CHW
            case 'SSSCB'
                % NOTE: Although fwd-PyTorch canonically uses the format "NCDHW",
                % we choose to preserve the order of the spatial dimensions,
                % treating them as HWD rather than DHW.
                permDLTToPython = [5 4 1 2 3]; % HWDCN -> NCHWD
            case 'SSSC'
                permDLTToPython = [4 1 2 3]; % HWDC -> CHWD
            case 'CB'
                permDLTToPython = [2 1]; % CN -> NC
            case 'SCBT'
                permDLTToPython = [3 2 4 1]; %SCBT -> BCTS
            otherwise 
                error(message('nnet_cnn_pytorchconverter:pytorchconverter:DlarrayFormatNotRecognized', labels));
        end
        dlX = permute(stripdims(dlX), permDLTToPython);
    else
        % Input is in reverse-PyTorch dimension order
        if rank<2
            % Do not permute rank 0 or 1 tensors.
            dlX = stripdims(dlX);
        else
            dlX = permute(stripdims(dlX), fliplr(1:rank));
        end
    end
end