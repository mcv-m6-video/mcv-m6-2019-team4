import numpy as np
import filterpy.kalman as pykalm

class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self, center):
        """Initialize variable used by Kalman Filter class (linear velocity)
        Args:
            None
        Return:
            None
        """
        self.dt = 0.005  # delta time

        self.M = np.array([[1, 0], [0, 1]])  # matrix in observation equations
        self.x = np.array(center)  # previous state vector

        # (x,y) tracking object center
        self.z = np.array([[0], [255]])  # vector of observations

        self.Sk = np.diag((3.0, 3.0))
        self.D = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat

        self.Sd = np.eye(self.x.shape[0])  # process noise matrix
        self.Sm = np.eye(self.z.shape[0])  # observation noise matrix
        self.lastResult = np.array([[0], [255]])
        #self.lastResult = np.array(center)

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            x: previous state vector
            Sk: previous covariance matrix
            D: state transition matrix
            Sd: process noise matrix
        Equations:
            x'_{k|k-1} = Dx'_{k-1|k-1}
            Sk_{k|k-1} = DSk_{k-1|k-1} D.T + Sd
            where,
                D.T is D transpose
        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.x = np.dot(self.D, self.x)
        # Predicted estimate covariance
        self.Sk = np.dot(self.D, np.dot(self.Sk, self.D.T)) + self.Sd
        self.lastResult = self.x  # same last predicted result
        return self.x

    def correct(self, z, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        x: predicted state vector x
        M: matrix in observation equations
        z: vector of observations
        Skm: predicted covariance matrix
        Sd: process noise matrix
        Sm: observation noise matrix
        Equations:
            C = AP_{k|k-1} M.T + R
            K_{k} = Sk_{k|k-1} M.T(C.Inv)
            x'_{k|k} = x'_{k|k-1} + K_{k}(b_{k} - Mx'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            z: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector x
        """

        if not flag:  # update using prediction
            self.z = self.lastResult
        else:  # update using detection
            self.z = z
        C = np.dot(self.M, np.dot(self.Sk, self.M.T)) + self.Sm
        K = np.dot(self.Sk, np.dot(self.M.T, np.linalg.pinv(C)))

        self.x = self.x + np.dot(K, (self.z - np.dot(self.M, self.x)))
        self.Sk = self.Sk - np.dot(K, np.dot(self.M, self.Sk))
        self.lastResult = self.x
        return self.x

class KalmanFilter_ConstantAcceleration(KalmanFilter):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self, center):
        """Initialize variable used by Kalman Filter class (linear velocity)
        Args:
            None
        Return:
            None
        """
        self.dt = 0.25  # delta time

        self.M = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])  # matrix in observation equations H
        # self.x = np.zeros((2, 1))  # previous state vector
        self.x = np.array([[center[0][0]], [center[1][0]], [0], [0], [0], [0]])  # previous state vector

        # (x,y) tracking object center
        # self.z = np.array([[0], [255]])  # vector of observations
        self.z = np.array([[center[0][0]], [center[1][0]]])  # vector of observations

        sigma = 0.1
        self.Sk = np.diag((sigma, sigma, sigma/10, sigma/10, sigma/100, sigma/100))  # covariance matrix P
        self.D = np.array([[1.0, 0.0, self.dt, 0.0,     self.dt**2/2, 0.0],
                           [0.0, 1.0, 0.0,     self.dt, 0.0,          self.dt**2/2],
                           [0.0, 0.0, 1.0,     0.0,     self.dt,      0.0],
                           [0.0, 0.0, 0.0,     1.0,     0.0,          self.dt],
                           [0.0, 0.0, 0.0,     0.0,     1.0,          0.0],
                           [0.0, 0.0, 0.0,     0.0,     0.0,          1.0]])  # state transition mat A



        sigmap = 1
        self.Sd = np.diag((sigmap, sigmap, sigmap/10, sigmap/10, sigmap/100, sigmap/100))  # covariance matrix P


        sigmax = 10
        self.Sm = np.array([[sigmax, 0.0],
                            [0.0, sigmax]])  # observation noise matrix R

        self.lastResult = np.array(np.array([[center[0][0]], [center[1][0]], [0], [0]]))

    def estimate_initial_velocity(self, center):
        """Initialize variable used by Kalman Filter class (linear velocity)
        Args:
            None
        Return:
            None
        """

        vx = (center[0][0]-self.x[0][0])/self.dt
        vy = (center[1][0] - self.x[1][0]) / self.dt
        self.x = np.array([[self.x[0][0]], [self.x[1][0]], [vx], [vy], [0], [0]])  # previous state vector


class KalmanFilter_ConstantVelocity(KalmanFilter):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self, center):
        """Initialize variable used by Kalman Filter class (linear velocity)
        Args:
            None
        Return:
            None
        """
        self.dt = 0.1  # delta time

        self.M = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])  # matrix in observation equations H
        # self.x = np.zeros((2, 1))  # previous state vector
        self.x = np.array([[center[0][0]], [center[1][0]], [0], [0]])  # previous state vector

        # (x,y) tracking object center
        # self.z = np.array([[0], [255]])  # vector of observations
        self.z = np.array([[center[0][0]], [center[1][0]]])  # vector of observations

        sigma = 0.1
        self.Sk = np.diag((sigma, sigma, sigma, sigma))  # covariance matrix P
        self.D = np.array([[1.0, 0.0, self.dt, 0.0],
                           [0.0, 1.0, 0.0, self.dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])  # state transition mat A


        sigmap = 1
        self.Sd = np.diag((sigmap, sigmap, sigmap/10, sigmap/10))  # covariance matrix P

        sigmax = 10
        self.Sm = np.array([[sigmax, 0.0],
                            [0.0, sigmax]])  # observation noise matrix R

        self.lastResult = np.array(np.array([[center[0][0]], [center[1][0]], [0], [0]]))

    def estimate_initial_velocity(self, center):
        """Initialize variable used by Kalman Filter class (linear velocity)
        Args:
            None
        Return:
            None
        """

        vx = (center[0][0]-self.x[0][0])/self.dt
        vy = (center[1][0] - self.x[1][0]) / self.dt
        self.x = np.array([[self.x[0][0]], [self.x[1][0]], [vx], [vy]])  # previous state vector

