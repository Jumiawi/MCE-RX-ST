function combined_features = fuse_feature_maps(swin_features, resnext_features, mcet_masks, alpha)
    combined_features = cell(size(swin_features));
    for i = 1:length(swin_features)
        combined_features{i} = alpha * (swin_features{i} .* mcet_masks{i}) + (1 - alpha) * resnext_features{i};
    end
end
