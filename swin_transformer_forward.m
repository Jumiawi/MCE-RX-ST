function feature_vector = swin_transformer_forward(image)
    patch_size = 4;
    embed_dim = 96;
    num_layers = 3;
    num_heads = 8;
    mlp_ratio = 4;
    epochs = 50;
    learning_rate = 0.001;
    feature_vector = swin_transformer_model(image, patch_size, embed_dim, num_layers, num_heads, mlp_ratio, epochs, learning_rate);
end
