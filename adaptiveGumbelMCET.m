function mcet_mask = adaptiveGumbelMCET(I)
    hn = imhist(I);
    L = length(hn);
    x = [];
    for i = 1:256
        x = [x, repelem(i, hn(i))];
    end
    skewness_value = skewness(double(x));
    if skewness_value < -0.5
        param = mingumbelmle(x);
        mu = param(1);
    elseif skewness_value > 0.5
        param = maxgumbelmle(x);
        mu = param(1);
    else
        mu = mean(x);
    end
    n = zeros(1, 255);
    for t = 1:255
        mu1 = mean(x(x <= t));
        mu2 = mean(x(x > t));
        n(t) = -sum((1:t) .* hn(1:t) * log(mu1)) - sum((t+1:L) .* hn(t+1:L) * log(mu2));
    end
    [D_min, threshold] = min(n);
    mcet_mask = I > threshold;
end
