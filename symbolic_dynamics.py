from pydrake import symbolic as sym
import numpy as np

class Kite:
    def __init__(self):
        self.m = 1      # kg
        self.g = 9.81   # m/s^2
        self.rho = 1.2  # air density
        self.A = 0.5    # kite surface area (m^2)
        self.C_l = 1.5  # coeff of lift
        self.C_d = 0.29 # coeff of drag

    def p(self, x):
        """ get the kite position in global frame
        """
        theta, phi, r, _, _, _ = x.ravel()

        s_t = sym.sin(theta)
        c_t = sym.cos(theta)
        s_p = sym.sin(phi)
        c_p = sym.cos(phi)

        x = r*s_t*c_p
        y = r*s_t*s_p
        z = r*c_t
        return np.array([x,y,z])

    def pdot(self, x):
        """ get the kite velocity in global frame
        """
        theta, phi, r, thetadot, phidot, rdot = x.ravel()

        s_t = sym.sin(theta)
        c_t = sym.cos(theta)
        s_p = sym.sin(phi)
        c_p = sym.cos(phi)

        xdot = r*(c_t*c_p*thetadot - s_t*s_p*phidot) + rdot*s_t*c_p
        ydot = r*(c_t*s_p*thetadot + s_t*c_p*phidot) + rdot*s_t*s_p
        zdot = r*(-s_t*thetadot) + rdot*c_t

        return np.array([xdot, ydot, zdot])

    def e_l(self, x, u, w):
        """ the unit vector along the kite longitudinal axis
        """
        w_e =  w - self.pdot(x)
        e_l = w_e/np.linalg.norm(w_e)
        return e_l

    def e_t(self, x, u, w):
        roll, _ = u.ravel()

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
        nu = sym.asin(np.dot(w_e, e_r)/w_p_e_norm*sym.tan(roll))

        # the basis in the plane of the kite pointing sideways (transverse)
        # accounting for kite roll
        e_t = e_w*(-sym.cos(roll)*sym.sin(nu)) +\
            e_o*(sym.cos(roll)*sym.cos(nu)) +\
            e_r*sym.sin(roll)

        return e_t

    def get_string_basis(self, x):
        """ get a coordinate frame rotated by the kite's position
        """
        theta, phi, _, _, _, _ = x.ravel()
        s_t = sym.sin(theta)
        c_t = sym.cos(theta)
        s_p = sym.sin(phi)
        c_p = sym.cos(phi)

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

    def f(self, x, u, w=np.array([6,0,0])):
        """ dynamics of the kite
        """
        theta, phi, r, thetadot, phidot, rdot = x.ravel()
        roll, torque = u.ravel()

        thetadotdot = self.F_theta_aer(x, u, w)/(r*self.m) +\
            sym.sin(theta)*self.g/r +\
            sym.sin(theta)*sym.cos(theta)*phidot**2 -\
            2*rdot*thetadot/r

        phidotdot = self.F_phi_aer(x, u, w)/(r*self.m) -\
            2/sym.tan(theta)*phidot*thetadot -\
            2*rdot*phidot/r

        rdotdot = torque/self.m +\
            r * thetadot**2 +\
            r * sym.sin(theta)**2 * phidot**2

        xdot = np.array([thetadot, phidot, rdot, thetadotdot, phidotdot, rdotdot])

        return xdot


if __name__ == '__main__':
    
    from pydrake.all import Variable

    q_theta = Variable('q_theta')
    q_phi = Variable('q_phi')
    q_r = Variable('q_r')
    qd_theta = Variable('qd_theta')
    qd_phi = Variable('qd_phi')
    qd_r = Variable('qd_r')
    u_roll = Variable('u_roll')
    u_tether = Variable('u_tether')

    x = np.ndarray([6,1], dtype=object,
        buffer=np.array([q_theta, q_phi, q_r, qd_theta ,qd_phi, qd_r]))
    u = np.ndarray([2,1], dtype=object,
        buffer=np.array([u_roll, u_tether]))

    kite = Kite()
    xdot = kite.f(x,u)
    for i, v in enumerate(x.ravel()):
        for j, vdot in enumerate(xdot[3:,].ravel()):
            print(i,j, vdot.Differentiate(v))
    # print(sym.Jacobian(xdot, x))


