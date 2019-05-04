#--------------------------------------------------------# 
#   Work 2 - Robot OpenAI using genetic algorithm
#   Student: Michael Canesche
#   ID: 68064
#   Prof.: Levi Lelis
#--------------------------------------------------------#

'''
	Libraries used
'''
import gym
import random
import matplotlib.pyplot as plt
import time
import sys
import numpy as np 
import copy

'''
	Global Constants
'''
MAX_GENERATIONS = 100 	# Max generation per epoch
ACTIONS_SIZE = 500	  	# Max actions have a individual
EPOCH_SIZE = 500		# Max epoch 		
GENE_INIT_SIZE = 4		# Size of Gene just began
GENE_SIZE = 16			# Size of Gene used on loop
ELITE_SIZE = 10			# Best chosen who will mutate 
HALL_FAME = 5			# Best of each generation
POPULATION_SIZE = 100 + ELITE_SIZE + HALL_FAME # Total of population
TOURNAMENT_SIZE = 10	# Size of tournament
PERCENT_MUTATION = 0.3 	# Probability of 30% to mutate

def population_start():
	'''
		Function: Population
		It is begin create the population
	'''
	return [[random.uniform(-1,1) for _ in range(4)] for _ in range(GENE_INIT_SIZE+GENE_SIZE)]

def crossover(p1, p2):
	'''
		Function: Crossover
		It is made the crossing between two individuals previously chosen
	'''
	c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

	c1_action_init = c1[:GENE_INIT_SIZE]
	c2_action_init = c2[:GENE_INIT_SIZE]

	part = random.randint(1, GENE_INIT_SIZE-1)

	c1_action_init[:part], c2_action_init[:part] = c2_action_init[:part], c1_action_init[:part]

	c1_action = c1[GENE_INIT_SIZE:]
	c2_action = c2[GENE_INIT_SIZE:]

	part = random.randint(1, GENE_SIZE-1)

	c1_action[:part], c2_action[:part] = c2_action[:part], c1_action[:part]

	c1 = c1_action_init + c1_action
	c2 = c2_action_init + c2_action

	return c1, c2


def mutation(ind):
	'''
		Function: Mutation
		For each gene, there is a chance for mutation
	'''
	for i in range(len(ind)):
		for j in range(len(ind[i])):
			n = random.random()
			if n < PERCENT_MUTATION :
				ind[i][j] += random.uniform(-0.5, 0.5)
	return ind

def save(ind, e):
	'''
		Function: save
		Responsible for saving the data of individual
	'''
	save_open = open("data/ind_best_epoch_"+str(e)+".txt", "w")
	save_open.write(str(ind[0])+" "+GENE_INIT_SIZE+" "+GENE_SIZE+"\n")
	for i in range(len(ind[1])):
		for j in range(len(ind[1][i])):
			save_open.write(str(ind[1][i][j]))
			if j != len(ind[1][i])-1:
				save_open.write(" ")
		save_open.write("\n")
	save_open.close()

def load(path):
	'''
		Function: load
		Responsible for loading the data of individual
	'''
	ind = []
	arq = open(path, "r")
	text = arq.readlines()
	i = 0
	lista = []
	for line in text:
		line = line.replace("\n", "").split(" ")
		if i == 0:
			global GENE_INIT_SIZE
			global GENE_SIZE
			GENE_INIT_SIZE, GENE_SIZE = int(line[1]), int(line[2])
			print(GENE_INIT_SIZE, GENE_SIZE)
			i = 1
			continue
		else:
			row = []
			for j in range(len(line)):
				row.append(float(line[j]))
			lista.append(row)
	arq.close()
	return lista

def evaluate(p, limit_steps=500, render=False):
	'''
		Function: evaluate or fitness
		Responsible for evaluating and obtaining the adaptation value.
	'''
	env = gym.make("BipedalWalker-v2")
	env.reset()
	score = 0

	print(GENE_INIT_SIZE, GENE_SIZE)

	#step begin, size 4 of p
	for i in range(GENE_INIT_SIZE):
		if render:
			env.render()
		#input(i)
		_, reward, _, _ = env.step(p[i])
		score += reward
	# loop until ACTION_SIZE
	i = 0
	cont = 0
	past_score = score
	for _ in range(GENE_INIT_SIZE, limit_steps):
		if render:
			env.render()
		_, reward, done, _ = env.step(p[(i % GENE_SIZE)+GENE_INIT_SIZE])
		score += reward

		if score < past_score:
			cont += 1

		if score < -100 or cont == 200:
			break
		past_score = score
		i += 1
	env.close()
	return score

def tournament(ind):
	'''
		Function: tournament
		Responsible for creating tournament to choose the best between them
	'''
	best_tournament = ind[random.randint(0,POPULATION_SIZE-1)]

	participants = random.sample(range(0,POPULATION_SIZE-1), TOURNAMENT_SIZE)

	for n in participants:
		if best_tournament[0] < ind[n][0]:
			best_tournament = ind[n]
	
	return best_tournament

def plot(value, e):
	'''
		Function: plot
		Responsible for plotting graphics and statistics
	'''
	plt.clf() # clean the plot
	plt.plot(value)
	plt.xlabel('Generations')
	plt.ylabel('Adaptation value')
	plt.savefig('figs/adaptation_epoch_'+str(e))
	
def main():
	'''
		Function: main
		Responsible for control all other functions
	'''
	for e in range(EPOCH_SIZE):

		print("epoch", e)

		begin = time.time()
		pop = []
		for i in range(2*POPULATION_SIZE):
			aux = population_start()
			pop.append([evaluate(aux), aux])
		time_spent = time.time()-begin

		best_global = pop[0]

		vector_fitness = []
		show = 10
		for i in range(MAX_GENERATIONS):

			begin = time.time()
			pop.sort(reverse=True, key=lambda ind: ind[0])

			best_individual = pop[0].copy()

			if best_individual[0] > best_global[0]:
				best_global = copy.deepcopy(best_individual)

			vector_fitness.append(best_individual[0])
			
			print("generation %2d: %10.6lf time spent: %.2fs" 
			%(i,best_individual[0], time_spent))
			
			if i == show:
				show += 10
				#evaluate(best_global[1], render=True)
			
			if i == 15:
				if best_global[0] < 0:
					break
			if i == 25:
				if best_global[0] < 2:
					break
			if i == 50:
				if best_global[0] < 10:
					break
			if i == 60:
				if best_global[0] < 15:
					break

			# fitness
			if best_global[0] >= 100 or i == MAX_GENERATIONS-1:
				
				print("melhor global =",best_global[0])

				save(best_global, e)
				
				plot(vector_fitness, e)
				
				break

			new_pop = []

			for j in range(HALL_FAME):
				aux = copy.deepcopy(pop[j][1])
				b1, b2 = evaluate(aux), evaluate(aux)
				new_pop.append([max(b1, b2), aux])

			for j in range(ELITE_SIZE):
				aux = mutation(copy.deepcopy(pop[j][1]))
				new_pop.append([evaluate(aux), aux])

			generate_children = (POPULATION_SIZE - ELITE_SIZE - HALL_FAME) // 2

			for j in range(generate_children):
				par1, par2 = tournament(copy.deepcopy(pop)), tournament(copy.deepcopy(pop))

				child1, child2 = crossover(par1[1], par2[1])
				eval1, eval2 = evaluate(child1), evaluate(child2)

				new_pop.append([eval1, child1])
				new_pop.append([eval2, child2])

			pop[:] = new_pop
			time_spent = time.time()-begin

if __name__ == "__main__":

	print(sys.argv)
	if len(sys.argv) == 2 :
		print("Executed file: " + sys.argv[1])
		evaluate(load(sys.argv[1]), render=True)
		exit(0)

	# Call main function
	main()