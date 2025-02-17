function [model, accuracy] = train_classifier(features, labels)
    model = fitcecoc(cell2mat(features), labels); % Train SVM model
    preds = predict(model, cell2mat(features));
    accuracy = sum(preds == labels) / numel(labels);
end
