function varargout = pyAtenNorm(varargin)
% Placeholder function for pyAtenNorm.

% Other parameters in `varargin{2:end}` are the parameters of the pytorch
% call the equivalent `torch.norm(descriptor, p=2, dim=1)`
inputTensor = varargin{1}; 
descriptor = inputTensor.value; %extractdata(inputTensor.value); % [1×256×60×80 dlarray] -> BxCxSxS

% Reimplement norm
sumSqDesc = sum(descriptor .^ 2, 2); % sum along `C`
channelNorm = sqrt(sumSqDesc); % sqrt of a 1x1x60x80 matrix
normalisedMatrix = descriptor ./ channelNorm;

% Assign the norm into varargout
result = struct('value', normalisedMatrix, 'rank', inputTensor.rank);
varargout{1} = result;
end