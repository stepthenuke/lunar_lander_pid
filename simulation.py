import gymnasium as gym
import numpy as np
import typing
import matplotlib.pyplot as plt

class Simulation():
    def __init__(self, coefficients: list[int|float] = None, contr_type: str = 'pid',
                 wind=True, turb_power=1, rend_mode='human'):
       
        self.environment = gym.make(
                'LunarLander-v2',
                continuous=True,
                enable_wind=wind,
                turbulence_power=turb_power,
                render_mode=rend_mode
                )

        if coefficients is not None:
            assert len(coefficients) == 6, \
                    'check lenght of coefficients'
            self.coef = coefficients
        else:    
            self.coef = [0. for _ in range(6)]

        assert all(ch in 'pid' for ch in contr_type), \
                'check control type'
        self.control_type = set(contr_type)

        self._history = []
        self.__integral_error = [0, 0]
        self.__reward = -300
     
    def control(self, coef=None, contr_type=None) -> np.ndarray:
        if contr_type is not None:
            assert all(ch in 'pid' for ch in contr_type), \
                    'check control type'
            control_type = set(contr_type)
        else:    
            control_type = self.control_type

        if coef is None:
            coef = self.coef
        else:
            assert len(coef) == 6, \
                    'check len of coef'
        
        state = self._history[-1]
        
        setpoint_alt = np.abs(state[0])
        setpoint_ang = 0.25 * np.pi * (state[0] + state[2])

        error_alt = setpoint_alt - state[1]
        error_ang = setpoint_ang - state[4]
        
        adjustment_alt, adjustment_ang = 0, 0

        if 'p' in control_type:
            adjustment_alt += coef[0] * error_alt
            adjustment_ang += coef[1] * error_ang
        
        if 'd' in control_type:
            adjustment_alt += coef[2] * state[3]
            adjustment_ang += coef[3] * state[5]

        if 'i' in control_type:
            self.__integral_error[0] += error_alt
            self.__integral_error[1] += error_ang
            adjustment_alt += coef[4] * self.__integral_error[0]
            adjustment_ang += coef[5] * self.__integral_error[1]
        
        action = np.array([adjustment_alt, adjustment_ang])
        action = np.clip(action, -1, 1)

        if state[6] or state[7]:
            action = np.array([0, adjustment_ang])
        return action

    def randomize_p_coef(self, level, scale=1):
        coef = np.array(self.coef[:2]).copy()
        coef += np.random.normal(0, 20.0 * scale / level, size=2)
        return list(coef)

    def randomize_d_coef(self,  level, scale=1):
        coef = np.array(self.coef[2:4]).copy()
        coef += np.random.normal(0, 20.0 * scale / level, size=2)
        return list(coef)

    def randomize_i_coef(self, level, scale=.001):
        coef = np.array(self.coef[4:]).copy()
        coef += np.random.normal(0, 20.0 * scale / level, size=2)
        return list(coef)
    
    def optimize(self, opt_type=None, max_level=100):
        if opt_type is not None:
            assert all(ch in 'pid' for ch in opt_type), \
                    'check control type'
            optimization_type = set(opt_type)
        else:    
            optimization_type = self.control_type
        
        self.reward = np.mean([self.run() for _ in range(5)])
        for lev in range(max_level):
            print(f'It: {lev}, reward: {self.reward}, coef: {self.coef}')
            self.optimize_random_step(optimization_type, lev+1)

        return self.coef

    def optimize_random_step(self, optimization_type, level):
        coef = self.coef.copy()
        if 'p' in optimization_type:
            coef[:2] = self.randomize_p_coef(level)
        if 'd' in optimization_type:
            coef[2:4] = self.randomize_d_coef(level)
        if 'i' in optimization_type:
            coef[4:] = self.randomize_i_coef(level)
        
        rewards = []
        for _ in range(5):
            reward = self.run(coef)
            rewards.append(reward)    
        avg_reward = np.mean(rewards)
        print(f'test reward: {avg_reward}, test coef: {coef}')
        if avg_reward > self.reward:
            print('swap')
            self.coef = coef
            self.reward = avg_reward
            

    def run(self, coef=None, contr_type=None):
        if coef is None:
            coef = self.coef
        self._history = []
        env = self.environment
        state, _ = env.reset()
        self._history.append(state)
        total_reward, self.__int_error = 0, [0,0]
        while True:
            action = self.control(coef, contr_type)
            state, reward, terminated, truncated, _ = env.step(action)
            self._history.append(state)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward    

    def graph(self):
        fig, ax = plt.subplots()
        history = np.array(self._history).T
        labels = ['x', 'y', 'vx', 'vy', 'angle', 'vangle']
        for var, lab in zip(history, labels):
            ax.plot(var, label=lab)
        ax.set(ylim=(-1.1, 1.1), title='PID control')
        ax.grid()
        ax.legend()
        plt.show()
