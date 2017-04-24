function [eigen_vectors,eigen_values] = limo_decomp(E,H)

% this function is used to decompose inv(E)*H
%
% Following Rencher 2002 (Methods of multivariate
% analysis - Wiley) we note that eig(inv(E)*H) =
% eig((E^1/2)*H*inv(E^1/2)) = eig(inv(U')*H*inv(U))
%
% E^1/2 is the square root matrix of E and U'U = E
% (Cholesky factorization). Using the Cholesky 
% factorisation, we return positve eigen values
% from inv(U')*H*inv(U) which is positive semidefinite.
% If this procedre fails (E is not positive definite) 
% we then use an SVD decomposition
%
% Cyril Pernet v2 29-05-2009
% -----------------------------
%  Copyright (C) LIMO Team 2010

try
    U = chol(E);
    [b D] = eig(inv(U')*H*inv(U)); % b: eigenvectors, D: diagonal matrix with eigenvalues
    
    % validate if correct eigenvalues and eigenvectors of matrix inv(U')*H*inv(U):
    if round((inv(U')*H*inv(U)) * b, 4) ~= round(b * D, 4) % needs to be equal 
        errordlg('something went wrong in the decomposition inv(U`)*H*inv(U)');
    else
        % adjustment to find eigenvectors of matrix inv(E)*H (page 279 Rencher) 
        a = inv(U)*b % these are the eigenvectors of inv(E)*H
        
        % sort eigenvectors 
        [e,ei] = sort(diag(D))
        a = a(:,flipud(ei)); % in order of increasing e
        
        % validate if correct eigenvalues and eigenvectors of matrix inv(E)*H:
        if round((inv(E)*H) * a, 4) == round(a * D, 4) % needs to be one (equal)
            % check if A * v = Eigenvalue * V
            % round((inv(E)*H) * a(:,1), 4) == round(eigen_values(1) * a(:,1), 4)
            eigen_vectors = a;
            eigen_values = flipud(e) % increasing order
        end   
            
    % check if (A - eigenvalue*I)*V = 0
    %((inv(E)*H) - (eigen_values(1) * eye(size(inv(E)*H, 1)))) * a(:,1)
    
    % vectors normalized? 
    % a(:,1)' * a(:,1) % if not equal to 1, not normalized. 
    
    end
      
catch
    y = (pinv(E)*H);
    [m n]   = size(y);
    if m > n
        [v s v] = svd(y*y');
        s       = diag(s);
        v       = v(:,1);
        u       = y*v/sqrt(s(1));
        eigen_vectors = v;
    else
        [u s u] = svd(y'*y);
        s       = diag(s);
        u       = u(:,1);
        v       = y'*u/sqrt(s(1));
        eigen_vectors = u;
    end
    d  = sign(sum(v)); u = u*d;
    eigen_values  = u*sqrt(s(1)/n);
end
