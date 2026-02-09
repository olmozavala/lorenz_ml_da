import numpy as np

class LorenzSystems:
    """
    Class to generate multiple types of Lorenz forms (63, 96, etc.).
    """
    
    @staticmethod
    def lorenz63(x, sigma=10.0, beta=8/3, rho=28.0):
        """
        Lorenz 63 system.
        """
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]
        return np.array([dx, dy, dz])

    @staticmethod
    def lorenz96(x, F=8.0):
        """
        Lorenz 96 system.
        """
        N = len(x)
        dxdt = np.zeros(N)
        for i in range(N):
            dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return dxdt

    @classmethod
    def get_system(cls, system_type):
        if system_type == '63':
            return cls.lorenz63
        elif system_type == '96':
            return cls.lorenz96
        else:
            raise ValueError(f"Unknown Lorenz system type: {system_type}")

    @classmethod
    def generate_trajectory(cls, system_type, x0, dt, n_steps, **params):
        """
        Generates a trajectory for a given Lorenz system.
        """
        f = cls.get_system(system_type)
        x0 = np.array(x0)
        nx = len(x0)
        trajectory = np.empty((n_steps, nx))
        trajectory[0] = x0
        x = x0.copy()
        
        for i in range(1, n_steps):
            x += dt * f(x, **params)
            trajectory[i] = x
            
        return trajectory
