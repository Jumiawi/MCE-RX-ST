function features = extract_resnext_features(images)
    features = cell(size(images));
    for i = 1:length(images)
        features{i} = resnext_forward(images{i});
    end
end
