function feature_vector = resnext_forward(image)
    num_blocks = 3;
    cardinality = 32;
    base_width = 4;
    epochs = 50;
    learning_rate = 0.001;
    feature_vector = resnext_model(image, num_blocks, cardinality, base_width, epochs, learning_rate);
end
