function Yout = pyElementwiseBinary(Xin1,Xin2,matlabFcn)
%Calculates elementise binary 'matlabFcn' for Xin1 and Xin2

%Copyright 2022 The MathWorks, Inc.

import traced_model2.ops.*

%'matlabFcn' can be plus, minus, div or mul
functionHandle = str2func(matlabFcn);

%Convert inputs to reverse pytorch format
Xin1Val = Xin1.value;
Xin2Val = Xin2.value;

Xin1Rank = Xin1.rank;
Xin2Rank = Xin2.rank;


if isdlarray(Xin1Val)
    [Xin1ValRevPyTorch, permRevPythonToDLT1] = permuteToReversePyTorch(Xin1Val);
else
    Xin1ValRevPyTorch = single(Xin1Val);
    permRevPythonToDLT1 = [];
end


if isdlarray(Xin2Val)
    [Xin2ValRevPyTorch, permRevPythonToDLT2] = permuteToReversePyTorch(Xin2Val);
else
    Xin2ValRevPyTorch = single(Xin2Val);
    permRevPythonToDLT2 = [];
end

if matlabFcn == "idivide"
    Xin1ValRevPyTorch = int32(floor(extractdata(Xin1ValRevPyTorch)));
    Xin2ValRevPyTorch = int32(floor(extractdata(Xin2ValRevPyTorch)));
end


YValRevPyTorch = functionHandle(Xin1ValRevPyTorch,Xin2ValRevPyTorch);


%Get labels and reverse permutation from the max rank input

%When rank is equal, we get the output label and reverse permutaion from
%the input that was in DLT format
%If none of the inputs were in DLT format, then the output will also be in
%reverse PyTorch format

if Xin1Rank == Xin2Rank
    Yrank = Xin1Rank;
    if ~isempty(permRevPythonToDLT1)
        revPerm = permRevPythonToDLT1;
        Ylabel  = dims(Xin1Val);
    elseif ~isempty(permRevPythonToDLT2)
        revPerm = permRevPythonToDLT2;
        Ylabel = dims(Xin2Val);
    else
        revPerm = [];
        Ylabel = dims(Xin1Val);
    end
else
    %When rank is not equal we get the permutation and label from the input
    %with max rank. If the revPerm for max rank input is empty then the
    %output is left in reverse python format.
    [Yrank, maxIndex] = max([Xin1Rank,Xin2Rank]);

    if maxIndex == 1
        revPerm = permRevPythonToDLT1;
        Ylabel = dims(Xin1Val);
    else
        revPerm = permRevPythonToDLT2;
        Ylabel  = dims(Xin2Val);
    end
end

%Empty revPerm implies that the input was already in Reverse Pytorch format
if ~isempty(revPerm)
    YVal = permute(YValRevPyTorch,revPerm);
    YVal = dlarray(single(YVal),Ylabel);
else
   YVal = dlarray(single(YValRevPyTorch),Ylabel);
end

Yout = struct("value",YVal,"rank",Yrank);

end
