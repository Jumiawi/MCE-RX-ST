function plot_confusion_matrix(conf_matrix)
    figure;
    imagesc(conf_matrix);
    colorbar;
    xlabel('Predicted');
    ylabel('True');
    title('Confusion Matrix');
end
