function [a, sampling_dist] = sample_from_unnormalized_log_prob(log_numerator)
%sample_from_unnormalized_log_prob(log_numerator)
%
% we want to sample from a discrete (categorical) distribution, but we only
% have a Kx1 vector of unnormalized log probabilities. furthermore, if we
% exponentiate this vector, many entries are numerically zero.
%
% this function draws a single sample from this distribution, using the
% "log sum exp" max trick to normalize the distribution even when the
% individual probabilities are tiny.

A = max(log_numerator);
log_denominator = A + log(sum(exp(log_numerator - A)));
log_sampling_dist = log_numerator - log_denominator;
sampling_dist = exp(log_sampling_dist);

[~, a] = max(rand() < cumsum(sampling_dist));