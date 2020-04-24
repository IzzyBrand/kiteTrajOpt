import numpy as np

class Kite:
    def __init__(self):
        self.r = 50.    # length of the cable in meters
        self.m = 1      # kg
        self.g = 9.81   # m/s^2
        self.rho = 1.2  # air density
        self.A = 0.5    # kite surface area (m^2)
        self.C_l = 1.5  # coeff of lift
        self.C_d = 0.29 # coeff of drag

    def p(self, x):
        """ get the kite position in global frame
        """
        theta, phi, _, _ = x

        s_t = np.sin(theta)
        c_t = np.cos(theta)
        s_p = np.sin(phi)
        c_p = np.cos(phi)

        x = self.r*s_t*c_p
        y = self.r*s_t*s_p
        z = self.r*c_t
        return np.array([x,y,z])

    def pdot(self, x):
        """ get the kite velocity in global frame
        """
        theta, phi, thetadot, phidot = x

        s_t = np.sin(theta)
        c_t = np.cos(theta)
        s_p = np.sin(phi)
        c_p = np.cos(phi)

        xdot = self.r*(c_t*c_p*thetadot - s_t*s_p*phidot)
        ydot = self.r*(c_t*s_p*thetadot + s_t*c_p*phidot)
        zdot = self.r*(-s_t*thetadot)

        return np.array([xdot, ydot, zdot])

    def e_l(self, x, u, w):
        """ the unit vector along the kite longitudinal axis
        """
        w_e =  w - self.pdot(x)
        e_l = w_e/np.linalg.norm(w_e)
        return e_l

    def e_t(self, x, u, w):
        # calculate the kite airspeed vector
        w_e =  w - self.pdot(x)

        # get the basis of the kite string
        e_theta, e_phi, e_r = self.get_string_basis(x)

        # project the wind into the plane of the kite
        w_p_e = w_e - e_r * np.dot(e_r, w_e)
        w_p_e_norm = np.linalg.norm(w_p_e)
        e_w = w_p_e/w_p_e_norm

        # get a vector perpendicular to the wind (sorta)
        # in the plane of the kite
        e_o = np.cross(e_r, e_w)

        # get the angle of the kite (defined by airspeed)????
        nu = np.arcsin(np.dot(w_e, e_r)/w_p_e_norm*np.tan(u))

        # the basis in the plane of the kite pointing sideways (transverse)
        # accounting for kite roll
        e_t = e_w*(-np.cos(u)*np.sin(nu)) +\
            e_o*(np.cos(u)*np.cos(nu)) +\
            e_r*np.sin(u)

        return e_t

    def get_string_basis(self, x):
        """ get a coordinate frame rotated by the kite's position
        """
        theta, phi, _, _ = x
        s_t = np.sin(theta)
        c_t = np.cos(theta)
        s_p = np.sin(phi)
        c_p = np.cos(phi)

        e_theta = np.array([c_t*c_p, c_t*s_p, -s_t])
        e_phi = np.array([-s_p, c_p, 0])
        e_r = np.array([s_t*c_p, s_t*s_p, c_t])
    
        return np.array([e_theta, e_phi, e_r])

    def F_theta_aer(self, x, u, w):
        """ the total aerodynamic force on azimuth
        """
        F_l = self.F_l(x, u, w)
        F_d = self.F_d(x, u, w)

        # get the basis of the kite string
        e_theta, e_phi, e_r = self.get_string_basis(x)

        e_l = self.e_l(x, u, w)
        e_t = self.e_t(x, u, w)

        F_theta_aer = F_l * (np.cross(e_l, e_t).T @ e_theta) +\
            F_d * (e_l.T @ e_theta)
        return F_theta_aer
    
    def F_phi_aer(self, x, u, w):
        """ the total aerodynamic force on elevation
        """
        F_l = self.F_l(x, u, w)
        F_d = self.F_d(x, u, w)

        # get the basis of the kite string
        e_theta, e_phi, e_r = self.get_string_basis(x)

        e_l = self.e_l(x, u, w)
        e_t = self.e_t(x, u, w)

        F_phi_aer = F_l * (np.cross(e_l, e_t).T @ e_phi) + F_d*(e_l.T @ e_phi)

        return F_phi_aer

    def F_l(self, x, u, w):
        """ force frome lift
        """
        w_e =  w - self.pdot(x)
        return 1/2 * self.rho * (w_e.T@w_e) * self.A * self.C_l

    def F_d(self, x, u, w):
        """ force from drag
        """
        w_e =  w - self.pdot(x)
        return 1/2 * self.rho * (w_e.T@w_e) * self.A * self.C_d

    def f(self, x, u, w=np.array([1,0,0])):
        """ dynamics of the kite
        """
        theta, phi, thetadot, phidot = x
    
        thetadotdot = self.F_theta_aer(x, u, w)/(self.r*self.m) +\
            np.sin(theta)*self.g/self.r +\
            np.sin(theta)*np.cos(theta)*phidot**2

        phidotdot = self.F_phi_aer(x, u, w)/(self.r*self.m) -\
            2/np.tan(theta)*phidot*thetadot

        xdot = np.array([thetadot, phidot, thetadotdot, phidotdot])

        return xdot