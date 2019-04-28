#--------------------------------------------------------# 
#   Work 2 - Robot OpenAI using genetic algorithm
#   Student: Michael Canesche
#   ID: 68064
#   Prof.: Levi Lelis
#--------------------------------------------------------#

import gym
import random

# constants
MAX_GENERATIONS = 30
number_actions = 500
percent_cross_over = 10 # e.g. 10 means 10% of total of actions
percent_mutation = 0

def get_action(env):
    return env.action_space.sample()

def get_n_worst(rewards, qtd):

	vector = []
	for i in range(len(rewards)):
		vector.append((i, rewards[i]))

	vector.sort(key=lambda tuple: tuple[1])
	#print(vector)
	
	result = []
	for i in range(qtd):
		result.append(vector[i][0])

	return result
    
# main function
def main():

	old_population = []

	env = gym.make("BipedalWalker-v2")

	for _ in range(number_actions):
		old_population.append([-100, get_action(env)])

	i = 0
	old_total = -100
	while i < MAX_GENERATIONS:
		print("generation %i:" %i)

		env.reset()

		arq = open("data/generation_"+str(i), "w")

		new_population = old_population.copy()
		new_total = 0.0

		#print(new_population)

		# It`s not necessary to do crossover on the first generations
		if i > 0 :
			get_actions_change = get_n_worst(
				[r[0] for r in new_population],
				number_actions*percent_cross_over//100)
			print(get_actions_change)
			for k in get_actions_change :
				new_population[k] = [-100, get_action(env)]

		for j in range(number_actions):
			
			env.render()

			action = new_population[j][1]
				
			#print('Action: ', action)
			observation, reward, done, info = env.step(action) # take a random action

			new_population[j][0] = reward

			arq.write(str(new_population[j][1])+"\n")
			arq.write(str(new_population[j][0])+"\n")

			if reward == -100 or j == number_actions-1 :
				print(new_total)
				if i == 0 :
					old_total = new_total
				elif new_total > old_total :
					# update the new population
					old_population = new_population.copy()
					old_total = new_total
					print("mudei o vetor old")
				break
			
			new_total += reward
			
		arq.close()

		i += 1

	env.close()


if __name__ == "__main__":

    # Call main function
    main()
