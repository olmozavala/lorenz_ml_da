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
            dxdt[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + F
        return dxdt

    @staticmethod
    def lorenz05(x, F=8.0, K=1):
        """
        Lorenz 2005 Model II system. 
        Generalization of Lorenz 96 with spatial smoothing.
        If K=1, it is identical to Lorenz 96.
        """
        N = len(x)
        dxdt = np.zeros(N)
        
        # Optimization for K=1 (Classic L96)
        if K == 1:
            for i in range(N):
                dxdt[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + F
            return dxdt

        # Model II smoothing parameter K
        J = K // 2
        if K % 2 == 0:
            w = np.ones(2*J + 1)
            w[0] = 0.5
            w[-1] = 0.5
        else:
            w = np.ones(2*J + 1)
        
        for n in range(N):
            val = 0
            for j_idx, j in enumerate(range(-J, J+1)):
                for l_idx, l in enumerate(range(-J, J+1)):
                    weight = w[j_idx] * w[l_idx]
                    t1 = -x[(n - 2 * K - l) % N] * x[(n - K - j + l) % N]
                    t2 = x[(n - K + j - l) % N] * x[(n + K + l) % N]
                    val += weight * (t1 + t2)
            
            dxdt[n] = val / (K * K) - x[n] + F
        return dxdt

    @classmethod
    def get_system(cls, system_type):
        if system_type == '63':
            return cls.lorenz63
        elif system_type == '96':
            return cls.lorenz96
        elif system_type == '05':
            return cls.lorenz05
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
