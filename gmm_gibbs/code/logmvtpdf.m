function [lp] = logmvtpdf(x, m, W, df)
% Compute the log likelihood of a multivariate student-t
% See Bishop page 692, appendix B
%
% x: observed vector, D x 1
% m: mean parameter, D x 1
% W: precision parameter, D x D
% df: degrees of freedom

lp = log(mvtpdf(x - m, W, df));
