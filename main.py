#--------------------------------------------------------# 
#   Work 2 - Robot OpenAI using genetic algorithm
#   Student: Michael Canesche
#   ID: 68064
#   Prof.: Levi Lelis
#--------------------------------------------------------#

import gym
import random
import matplotlib.pyplot as plt
import time
import sys
import numpy as np 

# constants global
MAX_GENERATIONS = 100
ACTIONS_SIZE = 500
GENE_SIZE = 10
POPULATION_SIZE = 100
ELITE_SIZE = 20
TOURNAMENT_SIZE = 10
PERCENT_MUTATION = 20 # means 20% 
EPSILON = 0.01
DELTA = 0 
env = gym.make("BipedalWalker-v2")

def population_start():
	ind = []
	for _ in range(2*GENE_SIZE):
		action = env.action_space.sample()
		ind.append(action)
	return ind

def crossover(p1, p2):

	p1 = np.matrix(p1)
	p2 = np.matrix(p2)

	part = random.randint(0, 4)

	init_p1x = p1[:GENE_SIZE,0:part]
	init_p2x = p2[:GENE_SIZE,part:4]

	p1x = p1[GENE_SIZE:2*GENE_SIZE,0:part]
	p2x = p2[GENE_SIZE:2*GENE_SIZE,part:4]

	part = random.randint(0, 4)

	init_p1y = p1[:GENE_SIZE,part:4]
	init_p2y = p2[:GENE_SIZE,0:part]

	p1y = p1[GENE_SIZE:2*GENE_SIZE,part:4]
	p2y = p2[GENE_SIZE:2*GENE_SIZE,0:part]

	aux_init_p1 = np.concatenate((init_p1x,init_p2x),axis=1)
	aux_init_p2 = np.concatenate((init_p1y,init_p2y),axis=1)

	aux_p1 = np.concatenate((p1x,p2x),axis=1)
	aux_p2 = np.concatenate((p1y,p2y),axis=1)

	c1 = list(np.array(aux_init_p1))+list(np.array(aux_p1))
	c2 = list(np.array(aux_init_p2))+list(np.array(aux_p2))

	return c1, c2

def mutation(p):

	for i in range(len(p)):
		n = random.randint(0, 100)
		#input(p[i])
		if n <= PERCENT_MUTATION :
			fator = EPSILON * (random.random() * 2-1)
			p[i] = p[i] + fator
			#print("passou",fator)
		#input(p[i]) 
		
	return p

def evaluate(p, render=False):
	env.reset()
	score = 0

	#step begin, size 10 of p
	for i in range(GENE_SIZE):
		if render:
			env.render()
		_, reward, _, _ = env.step(p[i])
		score += reward
	
	# loop until ACTION_SIZE
	i = 0
	for _ in range(GENE_SIZE, ACTIONS_SIZE):
		if render:
			env.render()
		_, reward, _, _ = env.step(p[(i % GENE_SIZE)+10])
		score += reward
		i += 1

	return score

def tournament(p):
	best_local = p[random.randint(0, POPULATION_SIZE-1)]

	for _ in range(TOURNAMENT_SIZE):
		n = random.randint(0, POPULATION_SIZE-1)
		if best_local[0] < p[n][0]:
			best_local = p[n].copy()
	
	return best_local

def plot(value):
	plt.plot(value)
	plt.xlabel('Generations')
	plt.ylabel('Adaptation value')
	plt.savefig('adaptacao')

# main function
def main():

	begin = time.time()
	pop = []
	for i in range(POPULATION_SIZE):
		aux = population_start()
		pop.append([evaluate(aux), aux])
	time_spent = time.time()-begin

	best_global = pop[0]

	value_before = pop[0][0]

	vector_fitness = []
	for i in range(MAX_GENERATIONS):

		begin = time.time()
		pop.sort(reverse=True, key=lambda ind: ind[0])

		best_individual = pop[0].copy()

		value_now = best_individual[0]

		DELTA = value_now - value_before
		value_before = value_now

		print("delta =",DELTA)

		# estrategia se perdi, eu quero eh mais
		if DELTA > 0 :
			PERCENT_MUTATION = 10
			EPSILON = 0.001
			ELITE_SIZE = 40
		else :
			PERCENT_MUTATION = 40
			EPSILON = 0.02
			ELITE_SIZE = 20

		if best_individual[0] > best_global[0]:
			best_global = best_individual.copy()

		vector_fitness.append(best_individual[0])
		
		print("generation %i: %10.6lf time spent: %.2fs" 
		%(i,best_individual[0], time_spent))

		# fitness
		if best_global[0] >= 100 or i == MAX_GENERATIONS-1:
			
			print("melhor global =",best_global[0])

			evaluate(best_global[1], render=True)

			env.close()
			
			plot(vector_fitness)
			
			break

		new_pop = []
		for j in range(ELITE_SIZE):
			aux = mutation(pop[j][1])
			new_pop.append([evaluate(aux), aux])

		generate_children = (POPULATION_SIZE - ELITE_SIZE) // 2

		for j in range(generate_children):
			par1 = tournament(pop)
			par2 = tournament(pop)

			child1, child2 = crossover(par1[1], par2[1])

			eval1 = evaluate(child1)
			eval2 = evaluate(child2)

			new_pop.append([eval1,child1])
			new_pop.append([eval2,child2])

		pop = new_pop.copy()
		time_spent = time.time()-begin
	
	env.close()

if __name__ == "__main__":

	if len(sys.argv) > 2 :
		print(sys.argv[1])

    # Call main function
	main()
