function Y = pyUnsqueeze(X, dim)
%PYUNSQUEEZE Inserts a singleton dimension at the position given by dim.
% at::Tensor at::unsqueeze(const at::Tensor &self, int64_t dim)

%   Copyright 2022 The MathWorks, Inc.

import traced_model2.ops.*

dim = dim.value;

% Convert the input data to reverse-Python dimension order
Xval = permuteToReversePyTorch(X.value);
Xrank = X.rank;

% Convert dim to reverse-Python dimension order
dim = Xrank - dim + 1;

% Reshape the data, inserting a singleton dim
Yrank = Xrank + 1;
if Yrank == 1
    newShape = size(Xval);
else
    newShape = ones(1, Yrank);
    knownSizes = setdiff(1:Yrank, dim);
    newShape(knownSizes) = size(Xval, 1:numel(knownSizes));
end
Yval = reshape(Xval, newShape);
Yval = dlarray(Yval, repmat('U', 1, Yrank));
Y = struct('value', Yval, 'rank', Yrank);
end