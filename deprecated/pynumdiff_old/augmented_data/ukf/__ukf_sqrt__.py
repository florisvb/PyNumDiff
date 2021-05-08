import numpy as np
import numpy.linalg as linalg
import scipy
import scipy.linalg

import oct2py
oct = oct2py.Oct2Py()
oct_cholupdate = oct.cholupdate

def cholupdate(A2, X, weight, language='python'):
    '''

    Downdate works, not sure why update (weight = +1) does not work. Use Octave for the update.

    Calculate chol(A + w x x')

    Parameters
    ----------
    A2 : [n_dim, n_dim] array
        A = A2.T.dot(A2) for A positive definite, symmetric
    X : [n_dim] or [n_vec, n_dim] array
        vector(s) to be used for x.  If X has 2 dimensions, then each row will be
        added in turn.
    weight : float
        weight to be multiplied to each x x'. If negative, will use
        sign(weight) * sqrt(abs(weight)) instead of sqrt(weight).

    Returns
    -------
    A2 : [n_dim, n_dim array]
        cholesky decomposition of updated matrix

    Notes
    -----

    Code based on the following MATLAB snippet taken from Wikipedia on
    August 14, 2012::

        function [L] = cholupdate(L,x)
            p = length(x);
            x = x';
            for k=1:p
                r = sqrt(L(k,k)^2 + x(k)^2);
                c = r / L(k, k);
                s = x(k) / L(k, k);
                L(k, k) = r;
                L(k,k+1:p) = (L(k,k+1:p) + s*x(k+1:p)) / c;
                x(k+1:p) = c*x(k+1:p) - s*L(k, k+1:p);
            end
        end
    '''
    # make copies

    X = np.array(X.copy())
    A2 = np.array(A2.copy())

    # standardize input shape
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    n_vec, n_dim = X.shape

    # take sign of weight into account
    sign, weight = np.sign(weight), np.sqrt(np.abs(weight))
    X = weight * X

    for i in range(n_vec):
        x = X[i, :]
        for k in range(n_dim):

            r = np.sqrt(A2[k,k]**2 + sign*x[k]**2)


            #r_squared = A2[k, k] ** 2 + sign * x[k] ** 2
            #r = 0.0 if r_squared < 0 else np.sqrt(r_squared)
            c = r / A2[k, k]
            s = x[k] / A2[k, k]
            A2[k, k] = r
            A2[k, k + 1:] = (A2[k, k + 1:] + sign * s * x[k + 1:]) / c
            x[k + 1:] = c * x[k + 1:] - s * A2[k, k + 1:]
    
    if language == 'python':
        return A2

    elif language == 'octave':
        if weight == -1:
            sgnW0 = '-'
        else:
            sgnW0 = '+'

        r = oct_cholupdate(A2, X.T, sgnW0)
        if type(r) != np.matrix:
            r = np.matrix(r)

        return r




def ukf_sqrt(y, u, x0, f, h, Q, R, alpha=0.01, beta=2):
    N = y.shape[1]

    nx = x0.shape[0]
    ny = y.shape[0]
    nq = Q.shape[0]
    nr = R.shape[0]

    a = alpha
    b = beta
    L = nx + nq + nr
    l = a**2*L - L
    g = np.sqrt(L + l)

    Wm = np.hstack(([[l/(L + l)]],  1/(2*(L + l))*np.ones([1, 2*L]))) # Weights for means
    Wm = np.matrix(Wm)
    Wc = np.hstack(([[(l/(L + l) + (1 - a**2 + b))]], 1/(2*(L + l))*np.ones([1, 2*L]) )) # Weights for covariances
    Wc = np.matrix(Wc)

    if Wc[0,0] >= 0:
        sgnW0 = 1
    else:
        sgnW0 = -1

    ix = np.arange(0, nx)
    iy = np.arange(0, ny)
    iq = np.arange(nx, (nx+nq))
    ir = np.arange((nx+nq), (nx+nq+nr))

    Sa = np.zeros([L,L])

    Sa[np.ix_(iq, iq)] = linalg.cholesky(Q).T
    Sa[np.ix_(ir, ir)] = linalg.cholesky(R).T

    Y = np.zeros([ny, 2*L+1]) # Measurements from propagated sigma points
    x = np.zeros([nx,N]) # Unscented state estimate
    P = np.zeros([nx,nx,N]) # Unscented estimated state covariance
    ex = np.zeros([nx, 2*L])
    ey = np.zeros([ny, 2*L])

    x[:,0:1] = x0
    P[:,:,0] = np.eye(nx)
    S = linalg.cholesky(P[:,:,0]).T

    for i in range(1, N):
        Sa[np.ix_(ix, ix)] = S

        # Only do this if R actually is time dependent
        # Sa[np.ix_(iq, iq)] = linalg.cholesky(Q[:,:,i]).T #chol(Q(:,:,i));
        # Sa[np.ix_(ir, ir)] = linalg.cholesky(R[:,:,i]).T #chol(R(:,:,i));


        xa = np.vstack([x[:,i-1:i], np.zeros([nq,1]), np.zeros([nr,1])])
        gsa = np.hstack((g*Sa.T, -g*Sa.T)) + xa*np.ones([1, 2*L])
        X = np.hstack([xa, gsa])

        # Propagate sigma points
        for j in range(0, 2*L+1):
            X[np.ix_(ix, [j])] = f(X[np.ix_(ix, [j])], 
                                   u[:,i-1:i], 
                                   X[np.ix_(iq, [j])])

            Y[:, j:j+1] = h(X[np.ix_(ix, [j])], 
                            u[:,i-1:i], 
                            X[np.ix_(ir, [j])])
            

        # Average propagated sigma points
        x[:,i:i+1] = X[np.ix_(ix, np.arange(0, X.shape[1]))]*Wm.T
        yf = Y*Wm.T

        # Calculate new covariances
        Pxy = np.zeros([nx,ny])
        for j in range(0, 2*L+1):
            ex[:,j:j+1] = np.sqrt(np.abs(Wc[0,j]))*(X[np.ix_(ix, [j])] - x[:,i:i+1])
            ey[:,j:j+1] = np.sqrt(np.abs(Wc[0,j]))*(Y[:,j:j+1] - yf)
            Pxy = Pxy + Wc[0,j]*(X[np.ix_(ix, [j])] - x[:,i:i+1])*(Y[:,j:j+1] - yf).T

        qr_Q, qr_R = scipy.linalg.qr( ex[:, 1:].T )
        S = cholupdate(qr_R[np.ix_(ix, ix)], ex[:, 0], sgnW0)

        # If no measurement at this time, skip the update step
        if any(np.isnan(y[:,i])):
            continue

        qr_Q, qr_R = scipy.linalg.qr( ey[:, 1:].T )
        Syy = cholupdate(qr_R[np.ix_(iy, iy)], ey[:, 0], sgnW0)
        

        # Update unscented estimate
        K = Pxy*np.linalg.pinv(Syy.T*Syy)
        x[:,i:i+1] = x[:,i:i+1] + K*(y[:,i:i+1] - h(x[:,i:i+1], u[:,i:i+1], np.zeros([nr,1])));
        U = K*Syy.T
        for j in range(ny):
            S = cholupdate(S, np.ravel(U[:,j]), 1, language='octave')

        P[:,:,i] = S.T*S
        
    s = np.zeros([nx,y.shape[1]]);
    for i in range(nx):
        s[i,:] = np.sqrt( P[i,i,:].squeeze() )
        
    return x, P, s