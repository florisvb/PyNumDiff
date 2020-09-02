%-----------------------------------------------------------------------
%dyneye
%Copyright (C) Floris van Breugel, 2013.
%  
%florisvb@gmail.com
%
%This function was originally written by Nathan Powell
%
%Released under the GNU GPL license, Version 3
%
%This file is part of dyneye.
%
%dyneye is free software: you can redistribute it and/or modify it
%under the terms of the GNU General Public License as published
%by the Free Software Foundation, either version 3 of the License, or
%(at your option) any later version.
%    
%dyneye is distributed in the hope that it will be useful, but WITHOUT
%ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
%FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
%License for more details.
%
%You should have received a copy of the GNU General Public
%License along with dyneye.  If not, see <http://www.gnu.org/licenses/>.
%
%------------------------------------------------------------------------

function [x, P, s] = ukf_sqrt(y, x0, f, h, Q, R, u)

N = length(y);      % Number of measurements

if size(Q,3) == 1
    Q = repmat(Q,[1,1,N]);
end

if size(R,3) == 1
    R = repmat(R,[1,1,N]);
end

nx = length(x0);
ny = size(y, 1);
nq = size(Q,1);
nr = size(R,1);

a = 1e-2;           % alpha
b = 2;              % beta
L = nx + nq + nr;   % L
l = a^2*L - L;      % lambda
g = sqrt(L + l);    % gamma

Wm = [l/(L + l) 1/(2*(L + l))*ones(1, 2*L)];                    % Weights for means
Wc = [(l/(L + l) + (1 - a^2 + b)) 1/(2*(L + l))*ones(1, 2*L)];  % Weights for covariances

if Wc(1) > 0
    sgnW0 = '+';
else
    sgnW0 = '-';
end

ix = 1:nx;
iy = 1:ny;
iq = nx+1:(nx+nq);
ir = (nx+nq+1):(nx+nq+nr);

% Construct initial augmented covariance estimate
Sa = zeros(L,L);
Sa(iq,iq) = chol(Q(:,:,1));
Sa(ir,ir) = chol(R(:,:,1));

% Pre-allocate
Y = zeros(ny, 2*L + 1); % Measurements from propagated sigma points
x = zeros(nx,N);        % Unscented state estimate
P = zeros(nx,nx,N);     % Unscented estimated state covariance
ex = zeros(nx, 2*L+1);
ey = zeros(ny, 2*L+1);

x(:,1) = x0;
P(:,:,1) = eye(nx);
S = chol(P(:,:,1));

for i = 2:N
    % Generate sigma points
    Sa(ix,ix) = S;
    
    % Only do this if R actually is time dependent
    Sa(iq,iq) = chol(Q(:,:,i));
    Sa(ir,ir) = chol(R(:,:,i));

    xa = [x(:,i-1); zeros(nq,1); zeros(nr,1)];
    X = [xa ([g*Sa' -g*Sa'] + xa*ones(1, 2*L))];
    
    % Propagate sigma points
    for j = 1:(2*L+1)
        X(ix,j) = f(X(ix,j), u(:,i-1), X(iq,j));
        Y(:,j) = h(X(ix,j), u(:,i-1), X(ir,j));
    end
    
    % Average propagated sigma points
    x(:,i) = X(ix,:)*Wm';   
    yf = Y*Wm';
    
    % Calculate new covariances
    Pxy = zeros(nx,ny);
    for j = 1:(2*L+1)
        ex(:,j) = sqrt(abs(Wc(j)))*(X(ix,j) - x(:,i));
        ey(:,j) = sqrt(abs(Wc(j)))*(Y(:,j) - yf);
        Pxy = Pxy + Wc(j)*(X(ix,j) - x(:,i))*(Y(:,j) - yf)';
    end
    
    [~, QR] = qr(ex(:,2:end)');
    fprintf('----------------')
    ex(:,1)
    fprintf('----------------')
    S = cholupdate(QR(ix,ix), ex(:,1), sgnW0);
    
    % If no measurement at this time, skip the update step
    if any(isnan(y(i)))
        continue
    end
    
    [~, QR] = qr(ey(:,2:end)');
    Syy = cholupdate(QR(iy,iy), ey(:,1), sgnW0);

    % Update unscented estimate
    K = Pxy/(Syy'*Syy);
    x(:,i) = x(:,i) + K*(y(i) - h(x(:,i), u(:,i), zeros(nr,1)));
    U = K*Syy';
    for j = 1:ny
        S = cholupdate(S, U(:,j), '-');
    end
    
    P(:,:,i) = S'*S;
end

s = zeros(nx,length(y));
for i = 1:nx
    s(i,:) = sqrt(squeeze(P(i,i,:)));
end

return
end
