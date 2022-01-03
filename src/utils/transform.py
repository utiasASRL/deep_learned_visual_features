"""Code to handle pose transforms. Some overlap with code in
   https://github.com/utiasASRL/vtr-dataset-tools/blob/master/transform.py.
"""

import numpy as np

class Transform:
    def __init__(self, C_ba, r_ab_inb):
        self.C_ba = C_ba
        self.r_ab_inb = r_ab_inb

    def __mul__(self, other):
        if other.__class__ == Transform:
            return Transform(self.C_ba.dot(other.C_ba), self.r_ab_inb + self.C_ba.dot(other.r_ab_inb))
        elif other.__class__ == np.ndarray:
            if other.shape in [(3, 1), (3,)]:
                return self.C_ba.dot(other) + self.r_ab_inb
            elif other.shape == (1, 3):
                return self.C_ba.dot(other.T) + self.r_ab_inb
            else:
                raise NotImplementedError("Cannot multiply with array of shape " + str(other.shape))
        else:
            raise NotImplementedError("Cannot multiply with type " + str(other.__class__))

    def inv(self):
        return Transform(self.C_ba.T, -self.C_ba.T.dot(self.r_ab_inb))

    @property
    def matrix(self):
        rval = np.eye(4)
        rval[:3, :3] = self.C_ba
        rval[:3, 3] = self.r_ab_inb

        return rval

    def __repr__(self):
        return str(self.matrix)

    @property
    def phi(self):
        # Get angle
        phi_ba = np.arccos(0.5*(self.C_ba.trace()-1.0))
        sinphi_ba = np.sin(phi_ba)

        if abs(sinphi_ba) > 1e-9:

            # General case, angle is NOT near 0, pi, or 2*pi
            axis = np.array([self.C_ba[2, 1] - self.C_ba[1, 2],
                             self.C_ba[0, 2] - self.C_ba[2, 0],
                             self.C_ba[1, 0] - self.C_ba[0, 1]])
            return (0.5*phi_ba/sinphi_ba)*axis

        elif abs(phi_ba) > 1e-9:
            # Angle is near pi or 2*pi
            # ** Note with this method we do not know the sign of 'phi', however since we know phi is
            #    close to pi or 2*pi, the sign is unimportant..

            # Find the eigenvalues and eigenvectors
            y, v = np.linalg.eig(self.C_ba)

            # Try each eigenvalue
            for i in range(3):
                # Check if eigen value is near +1.0
                if abs(y[i] - 1.0) < 1e-6:
                    # Get corresponding angle-axis
                    return phi_ba*v[i]

            # Runtime error
            raise RuntimeError("so3 logarithmic map failed to find an axis-angle, "
                               "angle was near pi, or 2*pi, but no eigenvalues were near 1")

        else:
            # Angle is near zero
            return np.zeros((3,))

    @staticmethod
    def hat(v):
        if len(v) == 4: v = v[:3] / v[3]
        skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
        return skv - skv.T
        # return np.array([[  0.0,  -v[2],   v[1]],
        #                  [ v[2],    0.0,  -v[0]],
        #                  [-v[1],   v[0],    0.0]])

    @staticmethod
    def SO3JacInv(aaxis_ba):
        ## separate the angle/axis
        angle_ba = np.linalg.norm(aaxis_ba)
        axis_ba = aaxis_ba / angle_ba

        # if its small then return identity
        if angle_ba < 1e-12:
            return np.eye(3, 3)

        halfphi = 0.5 * angle_ba
        cotanTerm = halfphi / np.tan(halfphi)

        ## jacinv = phi/2 * cot(phi/2)eye + (1-phi/2*cot(phi/2))aa^T - phi/2 * a^
        jacinv = np.eye(3, 3) * cotanTerm + (1.0 - cotanTerm) * \
                 axis_ba * axis_ba.transpose() - halfphi * Transform.hat(axis_ba)

        return jacinv

    @staticmethod
    def LogMap(tf):
        # Get the SO3 inverse jacobian
        jacinv = Transform.SO3JacInv(tf.phi)
        # multiply by translation to get rho
        rho_ab_inb = np.matmul(jacinv, tf.r_ab_inb.reshape((3, 1)))
        # return 6x1 vector
        return np.concatenate((rho_ab_inb.flatten(), tf.phi), axis=0)

    @classmethod
    def ExpMap(cls, xi):
        phi_ba = np.linalg.norm(xi[3:])
        if abs(phi_ba) < 1e-9:
            # If angle is very small, return identity rotation with linear offset
            return Transform(np.eye(3), xi[:3])

        axis = xi[3:] / phi_ba
        sin_term = np.sin(phi_ba) / phi_ba
        cos_term = (1.0 - np.cos(phi_ba)) / phi_ba

        A = cls.hat(axis)

        J = sin_term * np.eye(3) + (1.0 - sin_term) * np.outer(axis, axis) + cos_term * A
        return Transform(np.eye(3) + phi_ba * A.dot(J), J.dot(xi[:3]))