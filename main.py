#--------------------------------------------------------# 
#   Work 2 - Robot OpenAI using genetic algorithm
#   Student: Michael Canesche
#   ID: 68064
#   Prof.: Levi Lelis
#--------------------------------------------------------#

import gym

# class model of genetic algorithm
class Model(object):

    def __init__ (self):
        self.env = gym.make("BipedalWalker-v2")
        self.reward = -100

    def key_reward(self, DATA):
        #return DATA[]
        pass

    def get_action(self):
        
        return self.env.action_space.sample()
        '''
        population = []
        while len(population) != self.POPULATION_SIZE :
            self.env.reset()
            population.append(self.env.action_space.sample())
        
        DATA = []
        for pop in population :
            DATA.append(self.env.step(pop))

        maior = 0
        for i in range(1, len(DATA)):
            if DATA[maior][1] < DATA[i][1]:
                maior = i

        #if DATA[maior][1] > self.reward:
        #    self.reward = DATA[maior][1]

        observation, reward, done, info = DATA[maior]

        print(observation)
        print(reward)

        return population[0]
        '''

# main function
def main():
    
    m = Model()
    
    population = []

    for _ in range(500):
        population.append([-100, m.get_action()])

    i = 0
    while True :
        print("generation %i:" %i)

        env = gym.make("BipedalWalker-v2")
        env.reset()

        total = 0.0

        arq = open("data/generation_"+str(i), "w")
        for j in range(500):
            
            env.render()

            action = population[j][1]

            '''
            # try create the best population 
            if i > 0 :
                # try a new action, can be a little better
                if population[j][0] <= 0 :
                    action = m.get_action()
            '''

            print('Action: ', action)
            observation, reward, done, info = env.step(action) # take a random action
            
            # update your result
            print('Reward: ', reward)

            if reward == -100 :
                break

            if reward > population[j][0]:
                population[j] = [reward, action]

            arq.write(str(population[j][1])+"\n")
            arq.write(str(population[j][0])+"\n")

            total += reward

        arq.close()
        print("sum = ",total)

        i += 1

        env.close()


if __name__ == "__main__":

    # Call main function
    main()
