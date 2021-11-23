#Genetic algorithm code


import random
import math
import numpy as np


class DNA:

    def __init__(self, upper_bound,lower_bound, x): 
        self.fitness = 0
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.x=x
        self.up_len = len("{0:b}".format(upper_bound)) 
        self.binary_string=[]
        for i in range(0,len(x)):
            self.binary_string.append(self.encode(self.x[i]))

        
        self.ind_x=[]
        for i in range(0,len(x)):
            self.ind_x.append([*self.binary_string[i]])


        

        self.gLength_x=[]
        for i in range(0,len(x)):
            self.gLength_x.append(len(self.ind_x[i]))
        
        

        self.calculate_fitness() #the mathmatical function 

    def calculate_fitness(self):
        if min(self.x) < self.lower_bound or max(self.x) > self.upper_bound: #we don't accept if x < lower_bound. 
            self.fitness=float('inf')
        else:
            
            self.fitness = np.float_power(self.x[0],2)+np.float_power(self.x[1]-100,2)


            
    def encode(self, n):
        return format(n, '0' + str(self.up_len) + 'b') #binary format, by knowing the how many digit upper bound has, e.g. 1 becoms 0000001, since the upper bound is 100

    def decode(self, b):
        return int(b, 2) #convert a string of binary to integer e.g. int("1000", 2) is 8. 

    def list_to_decimal(self): # convert the list of binary ['0','0','1'] to integer e.g.  1
        for i in range(0,len(self.x)):
            self.x[i] = self.decode(bin(int(''.join(map(str, self.ind_x[i])), 2)))


    def update_child(self): 
        self.list_to_decimal()
        for i in range(0,len(self.x)):
            self.binary_string[i] = self.encode(self.x[i])


        self.calculate_fitness()

class Population: 

    def __init__(self, size, lower_bound, upper_bound,n_features):
        self.fittest = 0
        self.n_features=n_features
        self.DNAArray = [] #list containing DNA objects, length of the list is population size
        self.populationSize = size #the population size in each generation
        self.build_population(size, lower_bound, upper_bound, n_features)

    def build_population(self, size, lower_bound, upper_bound, n_features): #building popluation or DNAArray, each element in DNAArray is a DNA object, length of DNAArray is population size
        for i in range(0, size):
            random_list=[]
            for j in range(0,n_features):
                random_list.append(random.randint(lower_bound, upper_bound))
            self.DNAArray.append(DNA(upper_bound, lower_bound, random_list))

    def get_fittest(self): #find the fittest memeber in DNAArray
        max__fit = float('inf') #the value of mathmatical function, initilized as infiniti
        max_fit_i = 0 # max_fit_i shows which memeber in DNAArray is the fittest (has the minimum function value) 
        for i in range(0, len(self.DNAArray)):
            #print('value of fuction'+str(self.DNAArray[i].fitness))
            if max__fit >= self.DNAArray[i].fitness: #checking the function value is smaller or not for each element
                max__fit = self.DNAArray[i].fitness
                max_fit_i = i
        self.fittest = self.DNAArray[max_fit_i].fitness # the minimum value of the function in this generation
        #print('fitest'+str(self.fittest))
        return self.DNAArray[max_fit_i] #return the fittest memeber, the fittest memeber is a DNA object

    def get_fittest_i(self): # similiar get_fittest method, this method returns, the index of the fittest memeber in DNAArray list
        max__fit = float('inf')
        max_fit_i = 0
        for i in range(0, len(self.DNAArray)):
            if max__fit >= self.DNAArray[i].fitness:
                max__fit = self.DNAArray[i].fitness
                max_fit_i = i
        return max_fit_i

    def get_next_fittest(self): # get_next_fittest is not being used anywhere in the code! 
         #second fittest memeber in the generation
        max_fit = 0
        max_fit2 = 0
        for i in range(0, len(self.DNAArray)): # !include fittest and second fittest in crossover. I think there is an error here, it return second larget memeber, not the second smallet
            if self.DNAArray[i].fitness < self.DNAArray[max_fit].fitness \
                    and self.DNAArray[i] is not self.DNAArray[max_fit]: #skip the fittest
                max_fit2 = max_fit 
                max_fit = i
            elif self.DNAArray[i].fitness < self.DNAArray[max_fit2].fitness \
                    and self.DNAArray[i] is not self.DNAArray[max_fit2]: ###### updated by Reza
                max_fit2 = i #A backslash at the end of a line tells Python to extend the current logical line over across to the next physical line. 

        return self.DNAArray[max_fit2] #

    def get_least_fit(self): 
        min_fit_val = float('inf')
        min_fit_i = 0
        for i in range(0, len(self.DNAArray)): 
            if min_fit_val >= self.DNAArray[i].fitness: #I think it should be <=
                min_fit_val = self.DNAArray[i].fitness
                min_fit_i = i
        return min_fit_i

    def calc_fitness(self): # returns fittest memeber, a DNA object, in DNAAarray
        for i in range(0, len(self.DNAArray)):
            self.DNAArray[i].calculate_fitness()
        self.fittest = self.get_fittest()
        
        return self.fittest 

class Demo:
        #DNA is first class, popluation is a class that inhereted DNA class methods, Demos has access to population and DNA methods. 
    def __init__(self, size, lower_bound, upper_bound,n_features): #size of population, lowerbound, upperbound, number of features
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound #updated by Reza
        self.size=size
        self.n_features=n_features
        self.generation_count = 0 #counting the generations
        self.population = Population(size, lower_bound, upper_bound, n_features) #create a poplutaion object
        self.fittest = self.population.calc_fitness() #the DNA object of the fittest memeber
        self.new_pool=[]
        best_fittest=self.fittest.x #being use in the stopping criteria
        n_iteration=0 #if number of iteration reaches a certan number, stop the while loop
        print("Generation:", self.generation_count, " Fittest:", self.fittest.fitness, "x = ", self.fittest.x) #self.fittest.fitness first runs (self.fittest), then runes .fitness
        while True: # old stopping criteria: using number of generatoin self.generation_count < 1000:
            self.new_pool = [] #list of the current population
            self.generation_count += 1 #next generateion statrtes
            self.new_pool.clear() #clear the previous memebers from the list

            self.new_pool.append(self.fittest) #the DNA object of the fittest memeber from previous 
            for i in range(0,int(self.size/10+1)): #for each 10 population, we have random member

                random_list=[]
                for j in range(0,self.n_features):
                    random_list.append(random.randint(lower_bound, upper_bound))
                #print('random x: '+ str(DNA(upper_bound, lower_bound, random_list).x))
                self.new_pool.append(DNA(upper_bound, lower_bound, random_list))
        
            while True:

                
                a=[]
                for i in self.new_pool: 
                    a.append((i.x))

                child = self.crossover() #cross over between two parents, I think from the previous population (generation)
                




                mutate_rate = random.randint(1, 100) #random integer between 1 and 100, both 1 and 100 are included
                if mutate_rate <= 10: #5% chance of mutation of a child that is born by cross over
                    child = self.mutate(child)
                    
                if max(child.x) > self.upper_bound or min(child.x) < self.lower_bound:
                    # print('eeeeee') #the code may stock here.
                    continue
                
                if child.x in a:

                    continue

                
                    self.new_pool.append(child)
                if self.size == len(self.new_pool):
                    break



                self.new_pool.append(child) #append that child

                if self.size == len(self.new_pool):
                    break
                
            self.population.DNAArray.clear() #clearn the list of DNA objects for the next generation
            #for i in self.new_pool: print(i.x)
            self.population.DNAArray.extend(self.new_pool) #the current poplutaion is being replace the old population (generation)
            self.fittest = self.population.calc_fitness() #updating the fittest for this population
            n_iteration+=1
            print('best'+str(best_fittest))
            print('self'+str(self.fittest.x))
            if best_fittest != self.fittest.x:
                print("Generation:", self.generation_count, " Fittest:", self.fittest.fitness,  "x = ", self.fittest.x)
                best_fittest=self.fittest.x
                n_iteration=0
                print(n_iteration)

            if n_iteration==5: #if the fittest is not being updated for this number of generation
                break

    def output(self):
        print('Generation'+str(self.generation_count))
        final_result=self.fittest.x
        final_result.append(self.fittest.fitness)
        return final_result #self.generation_count,self.fittest.fitness,
        

    def crossover(self): #cross over of two DNA object, it is for point search
        #while True:


        partner_a = random.randint(0, self.population.populationSize - 1) #select a memeber from the (!previous) population randomly to be first parent
        partner_b = random.randint(0, self.population.populationSize - 1)
        while partner_a == partner_b: #parents should not be the same
            partner_b = random.randint(0, self.population.populationSize - 1)

        partner_a=self.population.DNAArray[partner_a]
        partner_b=self.population.DNAArray[partner_b]

        child = DNA(self.upper_bound, self.lower_bound, partner_a.x.copy()) #partner_a is DNA object, partner_a.num is the value of x e.g. 24
        child2 = DNA(self.upper_bound, self.lower_bound, partner_b.x.copy()) 
        for j in range(0,len(partner_a.x)):#len(partner_a.x) is the number of variables 
            
            crossover_point_x1_1 = random.randint(0, partner_a.gLength_x[j] - 2) #selecting the first digit between the first digit and (last digit-1)
            
            crossover_point_x1_2 = random.randint(crossover_point_x1_1+1, partner_a.gLength_x[j])  #updated by reza !!selecting the second digit between the first selected digit and last digit, I think X_1+1 should be, so don't include the first selected digit


            for i in range(crossover_point_x1_1, crossover_point_x1_2):
                child.ind_x[j][i] = str(partner_b.ind_x[j][i]) # first child is partner a, but get a binary digit from  partner b
                child2.ind_x[j][i] = str(partner_a.ind_x[j][i]) #is you don't use str here, the index will be assiging and you may end up updating the fittest member incorrectly


        child.update_child() #update the DNA object for the first child
        child2.update_child() #update the DNA object for the second child



        return child if child.fitness < child2.fitness else child2 #interesting

    def mutate(self, child): #gene or child is an object, mutation add randomness and it is for global search
        while True: #loop till the new generation is smaller than upper_bound
            parent_mutatation=random.randint(0, self.population.populationSize - 1)
            mutation_member= (child.x.copy()) 
            mutation_point1 = random.randint(1, child.gLength_x[0]-1) 
            mutation_point2 = random.randint(1, child.gLength_x[0]-1) # randomly select second mutuation point.
            if mutation_point1 == mutation_point2:
                continue #run the loop again
            
            mutation_point_list=[mutation_point1,mutation_point2]
            
            for mutation_point in mutation_point_list:
                for i in range(0,len(child.ind_x)):

                    #update all features (or all the Xs)
                    if child.ind_x[i][mutation_point] == 0: #change the binary format of the mutation point from 0 to 1 or from 1 to 0
                        child.ind_x[i][mutation_point] = 1
                    else:
                        child.ind_x[i][mutation_point] = 0
                    child.update_child() #update the DNA object

            
            break
        return child



population_size=10
upper_bound=100000 
lower_bound=0
N_features=2
# data rows of csv file  
optimum_x=[]
for i in range(0,1000):
    print('**********************')
    print('loop number= '+str(i))
    print('**********************')
    optimum_x.append(Demo(population_size,lower_bound, upper_bound,N_features).output())

import csv 
from csv import writer
# field names  
fields = ['x1', 'x2']  
# name of csv file  
filename = "test.csv"
# writing to csv file  
with open(filename, 'w', newline='') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)   
    csvwriter.writerows(optimum_x)