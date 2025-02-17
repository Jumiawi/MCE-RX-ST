function features = extract_swin_features(images)
    features = cell(size(images));
    for i = 1:length(images)
        features{i} = swin_transformer_forward(images{i});
    end
end
