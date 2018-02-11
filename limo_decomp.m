function [eigen_vectors,eigen_values] = limo_decomp(E,H,type)

% FORMAT: [eigen_vectors,eigen_values] = limo_decomp(E,H,type)
%
% INPUT E and H are matrices, typically square symmetric Sum of Squares and
%       Cross Products
%       type is 'Chol' (default) or 'SVD'
%
% OUTPUT: the eigen vectors and values of the decomposition of inv(E)*H
%
% Following Rencher 2002 (Methods of multivariate analysis - Wiley) we note
% that eig(inv(E)*H) = eig((E^1/2)*H*inv(E^1/2)) = eig(inv(U')*H*inv(U))
% and E^1/2 is the square root matrix of E and U'U = E (Cholesky factorization).
% Using the Cholesky factorisation, we return positve eigen values from
% inv(U')*H*inv(U) which is positive semidefinite. If this procedure fails
% (E is not positive definite) we then use an eigen value decomposition of pinv(E)*H
% It is also possible to procede using an SVD decomposition using the argument
% type ('SVD').
%
% Cyril Pernet 2009
% Cyril Pernet and Iege Bassez 2017
% -----------------------------
%  Copyright (C) LIMO Team 2010

% check input
if nargin < 2
    error('not enough arguments in')
elseif nargin == 2
    type = 'Chol';
end

% procede
if strcmpi(type,'chol')
    U = chol(E);
    [vec, D] = eig(inv(U')*H*inv(U)); % vec: eigenvectors, D: diagonal matrix with eigenvalues of inv(U')*H*inv(U) 

    % adjustment to find eigenvectors of matrix inv(E)*H (see page 279 Rencher 2002) 
    a = inv(U)*vec; % these are the eigenvectors of inv(E)*H

    % sort eigenvalues and then sort eigenvectors in order of decreasing eigenvalues
    [e,ei] = sort(diag(D));  % eigenvalues of inv(U')*H*inv(U) == eigenvalues of inv(E)*H
    ordered_eigenvalues = flipud(e);
    a = a(:,flipud(ei)); 

    % validate if correct eigenvalues and eigenvectors of matrix inv(E)*H
    % are returned:
    if round((inv(E)*H) * a, 4) == round(a * diag(ordered_eigenvalues), 4) 
        eigen_vectors = a;
        eigen_values = ordered_eigenvalues;
    
    else % if chol approach does not give the correct eigenvalues or eigenvectors (because E and/or H are singlar):
        [vec, D] = eig((pinv(E)*H));
        
        % sort eigenvalues and then sort eigenvectors in order of decreasing eigenvalues
        [e,ei] = sort(diag(D));  % eigenvalues of inv(U')*H*inv(U) == eigenvalues of inv(E)*H
        ordered_eigenvalues = flipud(e);
        vec = vec(:,flipud(ei));
        
        % validate if correct eigenvalues and eigenvectors of matrix pinv(E)*H:
        if round((pinv(E)*H) * vec, 4) == round(vec * diag(ordered_eigenvalues), 4)
            eigen_vectors = vec;
            eigen_values = ordered_eigenvalues;
        else
            error('this method could not find the correct eigenvalues or eigenvectors, try using SVD')
        end
    end   

% HAS TO BE CHANGED: 
elseif strcmpi(type,'SVD') % note: gives different/wrong eigenvalues than Rencher book.
    y = (pinv(E)*H);
    [m, n]   = size(y);
    if m > n
        [v,s,v] = svd(y*y');
        s       = diag(s);
        v       = v(:,1);
        u       = y*v/sqrt(s(1));
        eigen_vectors = v;
    else
        [u, s,u] = svd(y'*y);
        s       = diag(s);
        u       = u(:,1);
        v       = y'*u/sqrt(s(1));
        eigen_vectors = u; % only gives one eigenvector?
    end
    d  = sign(sum(v)); u = u*d;
    eigen_values  = u*sqrt(s(1)/n);
end