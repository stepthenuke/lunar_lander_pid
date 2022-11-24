import simulation as sm 

sim = sm.Simulation(wind=False, turb_power=0)
#coef = sim.optimize(opt_type='pd')
sim.coef = [1.6366748221682974, -4.147750351709235, -2.323712599579363, 2.6427835321523703, 0.0, 0.0]
sim.run()
sim.graph()
