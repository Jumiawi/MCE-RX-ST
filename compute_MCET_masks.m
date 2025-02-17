function masks = compute_MCET_masks(images)
    masks = cell(size(images));
    for i = 1:length(images)
        masks{i} = adaptiveGumbelMCET(images{i});
    end
end
