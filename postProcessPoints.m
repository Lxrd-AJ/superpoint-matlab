function [points, heatmaps] = postProcessPoints(pts, opts)
    % postProcessPoints 
    % Returns 
    %   `points` :  A 3xN matrix of points with the format [x_i, y_i, confidence]^T
    %   `heatmap`:  A HxW matrix in the range [0,1] of interest point confidence across entire image
    arguments
        %   pts: A dlarray with format SSCB
        %        For a given input image of size 480x640, `pts` should be `60(S)x80(S)x65(C)x1(B)`
        pts
        opts.DownsamplingFactor = 8
        opts.ConfidenceThreshold = 0.015
    end
    
    points = softmax(pts);

    % The 65 channels correspond to local, non-overlapping 8x8 grid region of pixels
    % plus an extra "no interest point" dustbin
    points = points(:,:,1:end-1,:); % This extra dustbin (last channel) is removed

    % --- HEATMAP ---
    % The network predicts interest points for every pixel in the image
    % This can be visualised as a heatmap
    % The heatmap can also be thresholded to derive useable interest points
    % `points` is of size `60(S)x80(S)x64(C)x1(B)`
    % First reshape to `60(S)x80(S)x64(C)x1(B)` -> `60(S)x80(S)x8(C)x8(C)x1(B)`
    sz = size(points);
    patchSize = opts.DownsamplingFactor;
    spatialDimensions = finddim(points, 'S'); %  e.g (1,2)
    s1 = spatialDimensions(1);
    s2 = spatialDimensions(2);
    batchSize = finddim(points, 'B'); % e.g 4
    % heatmap will be unformatted due to the `reshape` operator
    % `reshape` also strips trailing dimensions of size 1, therefore its need to adapt to arbitrary batch sizes
    heatmap = reshape(points, sz(s1), sz(s2), patchSize, patchSize, sz(batchSize));
    if ndims(heatmap) == 4 % Then the batch dimension has been stripped
        % TODO: Fix issue with unsqueeze `heatmap`
        heatmap = permute(heatmap, [1 2 3 4 5]);
    end

    % Map the heatmap to the original image dimensions i.e SxSxC
    % Permute the heatmap so that each spatial dimension has its `8x8` patch next to it
    heatmap = permute(heatmap, [1 3 2 4 5]); % SxCxSxCxB
    % Resize into original image size i.e H(S)xW(S)x1(C)xN(B)
    H = sz(1) * patchSize;
    W = sz(2) * patchSize;
    heatmaps = reshape(heatmap, H, W, 1, []);

    % -- POINTS --
    % Filtering
    filteredHeatmaps = heatmaps;
    filteredHeatmaps(filteredHeatmaps <= opts.ConfidenceThreshold) = 0;
    if ndims(filteredHeatmaps) == 4
        numInBatch = size(filteredHeatmaps, 4);
        % Extract keypoints for each batch
        points = cell(1,numInBatch);
        for k=1:numInBatch
            currentHeatmap = filteredHeatmaps(:,:,:,k);
            [yCoords, xCoords, ptConfidence] = find(extractdata(currentHeatmap));
            points{k} = horzcat(yCoords, xCoords, ptConfidence);
        end
    else
        [yCoords, xCoords, ptConfidence] = find(extractdata(filteredHeatmaps));
        points = horzcat(yCoords, xCoords, ptConfidence);
    end
end

function [suppressedPoints, pointIndices] = iNonMaximumSuppression(points, imageSize, opts)
    % iNonMaximumSuppression
    % Filter out interest points that are close to each based on the threshold  specified in `opts.nmsDistance`
    arguments
        points
        imageSize
        opts.nmsDistance = 4
    end
    % TODO: Continue
end