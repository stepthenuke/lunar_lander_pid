import simulation as sm 

sim = sm.Simulation(wind=False, turb_power=0)
coef = sim.optimize(opt_type='pd')

