"""
This file contains an implementation of online CGMM-MVDR. It is based 
on following papers:
[1] T. Higuchi, K. Kinoshita, N. Ito, S. Karita and T. Nakatani, "Frame-by-Frame Closed-Form Update for Mask-Based Adaptive MVDR Beamforming," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp. 531-535, doi: 10.1109/ICASSP.2018.8461850.
[2] T. Higuchi, N. Ito, S. Araki, T. Yoshioka, M. Delcroix and T. Nakatani, "Online MVDR Beamformer Based on Complex Gaussian Mixture Model With Spatial Prior for Noise Robust ASR," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 25, no. 4, pp. 780-793, April 2017, doi: 10.1109/TASLP.2017.2665341.

In docstrings, we use the following notation:
    C: number of channels
    F: number of frequency bins
"""

import numpy as np

def complex_gaussian(y, R, phi):
    '''Computes a zero-mean complex Gaussian distribution, Eq.(15) in [2].

    Arguments:
        y -- complex observation, shape (C, F)
        R -- parameter of the distribution, normalized covariance matrix, shape (F,C,C)
        phi -- parameter of the distribution, time-dependent variance, shape (F,)
               (phi * R is the covariance of the distribution)
               
    Returns:
       Evaluation of complex Gaussian at vector y, shape (F,)
    '''
    Sigma = phi[:,None,None] * R
    C = R.shape[1]
    ret = np.exp(-np.einsum('cf,fcd,df->f', 
                            y.conj(), np.linalg.inv(Sigma), y
                            )
                ) / np.abs(np.pi**C * np.linalg.det(Sigma))
    return ret.real.astype('float64')

class OnlineCGMMMVDR():
    def __init__(self, ny_k, ny_n, priorRk, priorRn, channels, frequency_bins,
                 beg_noise=0, end_noise=0):
        '''Initialization of online CGMM-MVDR beamforming.
        See [2], Figure 3 for a graphical model of the CGMM.
        In contrast with [2], we use different hyperparameter ny 
        for each component.

        Arguments:
            ny_k -- hyperparamater of CGMM prior on R, for target component, float.
            ny_n -- hyperparameter of CGMM prior on R, for noise component, float.
            priorRk -- a priori expectation of CGMM parameter R, for target comoponent.
                       shape (F, C, C)
            priorRk -- a priori expectation of CGMM parameter R, for noise comoponent.
                       shape (F, C, C)
            channels -- number of channels
            frequency_bins -- number of frequency bins
            beg_noise -- number of initial frames with noise only
                         masks are forced to be 0 at these frames
            end_noise -- number of final frames with noise only
                         masks are forced to be 0 at these frames
        '''
        self.ny_k = ny_k
        self.ny_n = ny_n

        self.C = channels
        self.F = frequency_bins
        self.beg_noise = beg_noise
        self.end_noise = end_noise

        # notation RR corresponds to \mathcal{R} in [2]
        self.RR_y_inv = np.tile(np.eye(self.C)[None], (self.F,1,1)) * 1e10
        self.RR_k = np.zeros((self.F,self.C,self.C), dtype = 'complex128')
        
        self.Lambda_kn = np.zeros((self.F,))
        self.Lambda_n = np.zeros((self.F,))

        self.R_kn = priorRk
        self.R_n = priorRn
        self.phi_kn = None
        self.phi_n = None
        self.alpha_kn = 0.5 * np.ones((self.F,))
        self.alpha_n = 0.5 * np.ones((self.F,))

    def _update_masks(self, y, is_all_noise=False):
        '''
        Estimates masks (posterior prob of d given obervation), Eq. (19),(25) in [2].

        If we have the extrenal knowledge of the frame containing noise only,
        the masks are forced to be 0.

        Arguments:
            y -- complex observation, shape (C,F)
            is_all_noise -- True if we know that the frame contains only noise

        Uses variables:
            self.R_kn -- normalized covariance matrix, parameter of cG for target component
                         shape (F,C,C)
            self.R_n -- normalized covariance matrix, parameter of cG for noise component
                         shape (F,C,C)
            self.phi_kn -- time-dependent variance, parameter of cG for target component
                           shape (F,)
            self.phi_n -- time-dependent variance, parameter of cG for noise component
                           shape (F,)
            self.alpha_kn -- mixture weight for target component, shape (F,)
            self.alpha_n -- mixture weight for noise component, shape (F,)

        Sets variables:
            self.lambda_kn -- mask for target component, shape (F,)
            self.lambda_n -- mask for noise components, shape (F,)
            self.Lambda_kn -- cummulation of the mask for target component, shape (F,)
            self.Lambda_n -- cummulation of the mask for noise component, shape (F,)
        '''
        p_y_kn = complex_gaussian(y, self.R_kn, self.phi_kn)
        p_y_n = complex_gaussian(y, self.R_n, self.phi_n)

        lambda_kn = self.alpha_kn * p_y_kn
        lambda_n = self.alpha_n * p_y_n
        self.lambda_kn = lambda_kn / (lambda_kn + lambda_n + 1e-6)
        self.lambda_n = lambda_n / (lambda_kn + lambda_n + 1e-6)
        if is_all_noise:
            self.lambda_kn = np.zeros_like(self.lambda_kn)
            self.lambda_n = np.ones_like(self.lambda_n)

        self.Lambda_kn += self.lambda_kn
        self.Lambda_n += self.lambda_n

    def _update_phi(self, y):
        '''
        Updates time-dependent variance, Eq.(20) in [2].

        Arguments:
            y -- complex observation, shape (C,F)

        Uses variables:
            self.R_kn -- normalized covariance matrix, parameter of cG for target component
                         shape (F,C,C)
            self.R_n -- normalized covariance matrix, parameter of cG for noise component
                         shape (F,C,C)

        Sets variables:
            self.phi_kn -- time-dependent variance, parameter of cG for target component
                           shape (F,)
            self.phi_n -- time-dependent variance, parameter of cG for noise component
                           shape (F,)
        '''
        self.phi_kn = np.einsum('cf,df,fdc->f', 
                                y, y.conj(), np.linalg.inv(self.R_kn)
                                ) / self.C
        self.phi_n = np.einsum('cf,df,fdc->f', 
                                y, y.conj(), np.linalg.inv(self.R_n)
                               ) / self.C

    def _update_R(self, y):
        '''
        Updates normalized covariance matrix, Eq. (33) in [2].

        Arguments:
            y -- complex observation, shape (C,F)

        Uses variables:
            self.lambda_kn -- mask for target component, shape (F,)
            self.lambda_n -- mask for noise components, shape (F,)
            self.Lambda_kn -- cummulation of the mask for target component, shape (F,)
            self.Lambda_n -- cummulation of the mask for noise component, shape (F,)
            self.phi_kn -- time-dependent variance, parameter of cG for target component
                           shape (F,)
            self.phi_n -- time-dependent variance, parameter of cG for noise component
                           shape (F,)

        Sets variables:
            self.R_kn -- normalized covariance matrix, parameter of cG for target component
                         shape (F,C,C)
            self.R_n -- normalized covariance matrix, parameter of cG for noise component
                         shape (F,C,C)
        '''
        # 1st summand in Eq (33)
        nom = self.Lambda_kn_prev + (self.ny_k + self.C + 1) / 2
        denom = self.Lambda_kn + (self.ny_k + self.C + 1) / 2
        R_kn_1 = (nom / denom)[:,None,None] * self.R_kn
        # 2nd summand in Eq (33)
        R_kn_2 = (1 / denom)[:,None,None] * np.einsum('f,f,cf,df->fcd', 
                                                      self.lambda_kn, 
                                                      1 / self.phi_kn, 
                                                      y, y.conj())
        self.R_kn = R_kn_1 + R_kn_2
        
        # 1st summand in Eq (33)
        nom = self.Lambda_n_prev + (self.ny_n + self.C + 1) / 2
        denom = self.Lambda_n + (self.ny_n + self.C + 1) / 2
        R_n_1 = (nom / denom)[:,None,None] * self.R_n
        # 2nd summand in Eq (33)
        R_n_2 = (1 / denom)[:,None,None] * np.einsum('f,f,cf,df->fcd', 
                                                     self.lambda_n, 
                                                     1 / self.phi_n, 
                                                     y, y.conj())
        self.R_n = R_n_1 + R_n_2

    def _update_RRk(self, y):
        '''
        Updates mask-based estimate of spatial covariance matrix of target signal.

        Equation (8) in [1].
        Scale factor is ignored as it does not matter in beamforming.

        Arguments:
            y -- complex observation, shape (C,F)

        Uses variables:
            self.lambda_kn -- mask for target component, shape (F,)

        Sets variables:
            self.RR_k -- spatial covariance matrix of target signal
                         shape (F,C,C)
        '''
        self.RR_k += np.einsum('f,cf,df->fcd', 
                                self.lambda_kn, y, y.conj()
                              )

    def _update_RRyinv(self, y):
        '''
        Updates inverse spatial covariance matrix of observed signal.

        Equation (7) in [1].
        Scale factor is ignored as it does not matter in beamforming.

        Arguments:
            y -- complex observation, shape (C,F)

        Sets variables:
            self.RR_y_inv -- spatial covariance matrix of observed signal
                             shape (F,C,C)
        '''
        nom = np.einsum('fab,bf,ef,feg->fag',
                        self.RR_y_inv,
                        y, y.conj(),
                        self.RR_y_inv)
        denom = 1 + np.einsum('cf,fcd,df->f', y.conj(), self.RR_y_inv, y)
        self.RR_y_inv = self.RR_y_inv - nom / denom[:,None,None]

    def _compute_beamformer(self, refs=(0,1)):
        '''
        Computes MVDR beamforming filters, Eq. (9) in [1].

        Computes two different filters with different reference microphones.

        Arguments:
            refs -- indexes of two reference microphones

        Uses variables:
            self.RR_y_inv -- spatial covariance matrix of observed signal
                             shape (F,C,C)
            self.RR_k -- spatial covariance matrix of target signal
                         shape (F,C,C)
        Returns:
            w_k1 -- beamforming filters corresponding to 1st ref microphone, shape (F,C)
            w_k2 -- beamforming filters corresponding to 2nd ref microphone, shape (F,C)
        '''
        ref1, ref2 = refs
        num1 = np.einsum('fbd,fdc->fbc', self.RR_y_inv, self.RR_k)[:,:,ref1]
        num2 = np.einsum('fbd,fdc->fbc', self.RR_y_inv, self.RR_k)[:,:,ref2]
        denom = np.einsum('fcd,fdc->f', self.RR_y_inv, self.RR_k)
        w_k1 = num1 / (denom[:,None] + 1e-10)
        w_k2 = num2 / (denom[:,None] + 1e-10)
        return w_k1, w_k2

    def _beamform(self, y, w_k):
        '''
        Performs beamforming with two beamforming filters.

        Arguments:
            w_k -- tuple of two beamforming filter coefficients, each of shape (F,C)
            y -- complex observation, shape (C,F)

        Returns:
            s_k1 -- beamformed signal with 1st beamformer, shape (F,)
            s_k2 -- beamformed signal with 2nd beamformer, shape (F,)
        '''
        w_k1, w_k2 = w_k
        s_k1 = np.einsum('fc,cf->f', w_k1.conj(), y)
        s_k2 = np.einsum('fc,cf->f', w_k2.conj(), y)
        return s_k1, s_k2

    def step(self, y, l, T):
        '''
        Performs one step of the inference and beamforming.

        Arguments:
            y -- complex observation, shape (C,F)
            l -- index of frame in the utterance (used to identify noise parts)
            T -- total number of frames in the utterance (used to identify noise parts)

        Returns:
            (s_k1, s_k2) -- beamformed signal using two beamformers, each shape (F,)
            lambda_kn -- estimated mask corresponsing to target signal, shape (F,)
        '''
        # Initialization of phi
        if self.phi_kn is None:
            assert self.phi_n is None
            self.phi_kn = np.einsum('cf,df,fdc->f', 
                               y, y.conj(), np.linalg.inv(self.R_kn)
                              ) / self.C
            self.phi_n = np.einsum('cf,df,fdc->f', 
                              y, y.conj(), np.linalg.inv(self.R_n)
                             ) / self.C

        self.Lambda_kn_prev, self.Lambda_n_prev = self.Lambda_kn, self.Lambda_n
        is_all_noise = (l < self.beg_noise) or (l > T-self.end_noise)
        self._update_masks(y, is_all_noise)
        self._update_phi(y)
        self._update_R(y)
        self._update_RRk(y)
        self._update_RRyinv(y)
        if l > self.beg_noise - 1:
            w_k = self._compute_beamformer()
            s_k1, s_k2 = self._beamform(y, w_k)
        else:
            s_k1, s_k2 = np.zeros((self.F,)), np.zeros((self.F,))

        return (s_k1, s_k2), self.lambda_kn
