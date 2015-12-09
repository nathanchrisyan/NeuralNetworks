#Evolutionary Simulation of Neural Networks

import math
import random
import pygame
import os

pygame.init()

BACKGROUND = (255, 255, 255)
(SCREENBOUNDX, SCREENBOUNDY) = (1300, 800)
screen = pygame.display.set_mode((SCREENBOUNDX, SCREENBOUNDY))

Font = pygame.font.SysFont("Comic Sans MS", 10)

Food = [] #The food pool
Agents = [] #The agent pool
AgentCounter = 1 #The amount of agents ever created
Species = []
TotalFitness = 0 #The total fitness of all the agent
PreviousFitness = 0 
MostFood = 0 #The most food any one agent has collected

toUpdate = []
Update = True #Are we updating the simulation?
WhenToUpdate = 1 #Changes the speed of the simulation
Ticks = 0 #The clock of the simulation
Quarter = 0
Select = 0 #The selected agent
SpeciesSelect = False #The selected species 

#Initialization of the food 
for i in range (5):
	Food.append(Food[5, random.randint(0, SCREENBOUNDX), random.randint(0, SCREENBOUNDY), (225, 205, 0)])

Clock = pygame.time.Clock() #FPS 

class Agent:
	def __init__(self, Input, Hidden, Output, WeightsIH, WeightsHO, Color, species,posx = random.randint(0, SCREENBOUNDX), posy = random.randint(0, SCREENBOUNDY)):
		self.Input = Input #The Input nodess
		self.Hidden = Hidden #The Hidden nodes 
		self.Output = Output #The Output nodes
		self.WeightsIH = WeightsIH #The weights from Input to Hidden
		self.WeightsHO = WeightsHO #The weights from Hidden to Output
		self.WeightBIAS1 = random.uniform(-1, 1)
		self.WeightBIAS2 = random.uniform(-1, 1)
		self.posx = posx #The y coordinate of the agent
		self.posy = posy #The x coordinate of the agent
		self.LookAtx = 0 #The look at vector of the agent
		self.LookAty = 0 #The look at vector of the agent
		self.Rotation = 0
		self.Health = 4000
		self.Clock = 0
		self.FoodColl = 0
		self.Color = Color
		self.fitness = 0
		self.species = species
	def createNeuralNetwork(self): #This function creates the neural networks of each individual agent, since all connections are implied, only the weights need to be assigned
		for i in range (len(self.Input)):
			for j in range(len(self.Hidden)):
				self.WeightsIH.append(random.uniform(-4,4))
		for k in range (len(self.Hidden)):
			for l in range (len(self.Output)):
				self.WeightsHO.append(random.uniform(-4,4))

class Food:
	def __init__(self, Size, posx, posy, color):
		self.Size = Size
		self.posx = posx 
		self.posy = posy
		self.color = color
		
def Clear():
	os.system("clear") 

def Initialize(agents): #This function initalizes all of the agents, the parameter agents defines how many agents there will be
	global AgentCounter
	for i in range(agents):
		AgentCounter += 1
		Agents.append(Agent([0,0,0,0,0], [0,0,0], [0,0,0],[],[], (random.randint(0, 225), random.randint(0, 225), random.randint(0, 225)), AgentCounter))
		Agents[i].createNeuralNetwork() #Initialize the agent then create its neural network


def AddFood(): #Adds a food randomly 
	Food.append(Food[5, random.randint(0, SCREENBOUNDX), random.randint(0, SCREENBOUNDY), (225, 205 ,0)])
	
def Add(): #Randomly adds a random agent 
	global AgentCounter
	AgentCounter += 1
	Agents.append(Agent([0,0,0,0,0], [0,0,0], [0,0,0],[],[], (random.randint(0, 225), random.randint(0, 225), random.randint(0, 225)), AgentCounter))
	Agents[len(Agents) - 1].createNeuralNetwork()	

	
def Clone(agent): #Clones an agent
	global AgentCounter
	AgentCounter += 1
	Agents.append(Agent([0,0,0,0,0], [0,0,0], [0,0,0], agent.WeightsIH, agent.WeightsHO, agent.Color, agent.posx, agent.posy, agent.species))
	
def Replicate(agent): #Clones the agent and mutates it
	global AgentCounter
	AgentCounter += 1	
	Agents.append(Agent([0,0,0,0,0], [0,0,0], [0,0,0], agent.WeightsIH, agent.WeightsHO, agent.Color, agent.posx, agent.posy, agent.species))
	PushChromosome(Mutate(Chromosome(Agents[len(Agents) - 1].WeightsIH, Agents[len(Agents)- 1].WeightsHO)), Agents[len(Agents)-1])
		
def Sigmoid(activation, p = 1.0): #The sigmoid function calculates the outputs of the neurons based on the activation
	return 1/(1 + math.e**(-activation /p))

def CalcDistance(Pos1, Pos2):
	return math.sqrt((Pos1[0]-Pos2[0]) ** 2 + (Pos1[1] - Pos2[1]) ** 2)
	
def ClosestFood(AgentPosx, AgentPosy, Food): #Finds the closest food in the food pool 
	ClosestDistance = 99999999
	ClosestIndex = 99999999
	for i in range (len(Food)):
		if ((AgentPosx - Food[i][0]) ** 2 + (AgentPosy - Food[i][1]) ** 2) < ClosestDistance:
			ClosestIndex = i
			ClosestDistance = ((AgentPosx - Food[i][0]) ** 2 + (AgentPosy - Food[i][1]) ** 2)
	return ClosestIndex
		
def Clamp(Num, min, max): #Clamps a variable, so it never goes above or below the max and min values respectively 
	if Num > max:
		Num = max
	if Num < min:
		Num = min
	return Num
	
def Normalise(vector): #Normalise function
	v = math.sqrt((vector[0]**2 + vector[1]**2))
	
	return [vector[0]/v, vector[1]/v]	

def Roulette(TotalFitness, Agent): #Picks agents out of the agent pool based on their fitness
	slice = random.randint(0, TotalFitness)
	
	CurrentSlice = 0
	Choose = 0
	
	for i in range(len(Agent)):
		CurrentSlice += Agent[i].fitness
		if CurrentSlice > slice:
			Choose = i
			break;
	return Choose
	
def Chromosome(Hidden, Output): #Copies the chromosome from an agent into a usable array 
	Chromosome = []
	for i in range(len(Hidden)):
		Chromosome.append(Hidden[i])
	for i in range(len(Output)):
		Chromosome.append(Output[i])
	return Chromosome

def PushChromosome(Chromosome, Agent): #Pushes the chromosome into an empty agent
	Hidden = []

	for i in range(len(Agent.WeightsIH)):
		
		Hidden.append(Chromosome[i])
	Output = []
	for j in range(len(Agent.WeightsHO), len(Chromosome)):
		Output.append(Chromosome[j])
	Agent.WeightsIH = Hidden
	Agent.WeightsHO = Output
	return Agent

def FindBestFitness(Agents): #Finds the agent with the best fitness
	CurrentBestFitness = -999999999
	Index = 0
	for i in range(len(Agents)):
		if Agents[i].fitness > CurrentBestFitness:
			CurrentBestFitness = Agents[i].fitness
			Index = i
	return Index

def Mutate(Chromosome, MutationRate = 0.1): #Mutation function 
	for i in range(len(Chromosome)):
		Mutate = random.random()
		if Mutate < MutationRate:
			Chromosome[i] += random.uniform(-1, 1) * random.random()

	return Chromosome


Initialize(10) #Initialize 10 agents
print AgentCounter
Generation = 0
Running = True

while Running:
	#Checks for key presses, for simulation interaction
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			Running = False 
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_d:
				Update = (True, False)[Update]
			elif event.key == pygame.K_s:
				WhenToUpdate += 1
			elif event.key == pygame.K_z:
				WhenToUpdate -= 1
			elif event.key == pygame.K_a:
				AddFood()
			elif event.key == pygame.K_LEFT:
				Select += 1
				if Select > len(Agents) - 1:
					Select = len(Agents) - 1
			elif event.key == pygame.K_RIGHT:
				Select -= 1
				if Select < 0:
					Select = 0
			elif event.key == pygame.K_q:
				SpeciesSelect = (True, False)[SpeciesSelect]
			elif event.key == pygame.K_r:
				Agents = []
				Food = [] #The food pool
				Agents = [] #The agent pool
				AgentCounter = 1 #The amount of agents ever created
				Species = []
				TotalFitness = 0 #The total fitness of all the agent
				PreviousFitness = 0 
				MostFood = 0 #The most food any one agent has collected

				toUpdate = []
				Update = True #Are we updating the simulation?
				WhenToUpdate = 1 #Changes the speed of the simulation
				Ticks = 0 #The clock of the simulation
				Quarter = 0
				Select = 0 #The selected agent
				SpeciesSelect = False #The selected species 
				Generation = 0
				Initialize(10)
	
	#FPS
	Clock.tick()
	pygame.display.set_caption(str(Clock.get_fps()))
	
	Ticks += 1 #The clock of the simulation 
	if len(Food) <= 0: #Checks if there is zero food in the environment
		for i in range(5):
			AddFood()
	
	#Adds food every 100 simulation ticks 
	if Ticks % 50 == 0:
		AddFood()
	
	if Ticks % 100 == 0:
		Clear()
		print "Epoch: ", Generation, "     ", "Simulation Ticks: ", Ticks 
		print TotalFitness/len(Agents), "Average Fitness     ", PreviousFitness, "Previous Fitness"
		print "Agents: ", len(Agents)
		print "---------------------------------------------------------"
		print " "
		
	if Ticks % 2000 == 0:
		Add()
	
	#Crossover and bisexual reproduction	
	if Ticks % 1000 == 0:
	
		#Print out statistics for that generation
		Clear()
		print "Epoch: ", Generation, "     ", "Simulation Ticks: ", Ticks 
		print TotalFitness/len(Agents), "Average Fitness     ", PreviousFitness, "Previous Fitness"
		print "Agents: ", len(Agents)
		print "---------------------------------------------------------"
		print " "
		PreviousFitness = TotalFitness/len(Agents)
		
		Best = FindBestFitness(Agents) #This bit creates more of the best agents, to maximize their chances of being selected for bisexual 									reproduction
		Agents[Best].fitness *= 3
		TotalFitness += (Agents[Best].fitness * 3) - Agents[Best].fitness
		TotalFitness = Clamp(TotalFitness, 1, 10000000000)
			
		Generation += 1
		for i in range(3):
			MumNum = Roulette(TotalFitness, Agents) #Selecting the agents 
			DadNum = Roulette(TotalFitness, Agents)
			
			Mum = Agents[MumNum]
			Dad = Agents[DadNum]
			
			Agents[DadNum].species = Agents[MumNum].species
			
			Mum2 = Chromosome(Mum.WeightsIH, Mum.WeightsHO) #Stripping their chromosomes 
			Dad2 = Chromosome(Dad.WeightsIH, Dad.WeightsHO)
			
			#Crossover
			child1 = [] #Creating the children chromosomes
			child2 = []

			for p in range(len(Mum2)): #Filling the children chromosomes 
				child1.append(0)
				child2.append(0)
			
			Agents.append(Agent([0,0,0,0,0], [0,0,0], [0,0,0],[], [], (Mum.Color), Mum.species))
			Agents[len(Agents) - 1].createNeuralNetwork() #Creating the children
			
			Agents.append(Agent([0,0,0,0,0], [0,0,0], [0,0,0], [], [],  (Mum.Color), Mum.species))
			Agents[len(Agents) - 1].createNeuralNetwork()
			
			
			CrossoverPoint = random.randint(0, 24) #Selecting the crossover point for the reproduction 
			for k in range(CrossoverPoint):  #Selecting genes from both the mother and the father
				child1[k] = Mum2[k]
				child2[k] = Dad2[k]
			for j in range(CrossoverPoint, 24):
				child1[j] = Dad2[j]
				child2[j] = Mum2[j]
			
			child1 = Mutate(child1, 0.1) #Mutating the children
			child2 = Mutate(child2, 0.1)
			
			PushChromosome(child1, Agents[len(Agents) - 1]) #Pushing the chromosomes into the children agents
			PushChromosome(child2, Agents[len(Agents) - 2])
			
			Best = FindBestFitness(Agents) 
			Agents[Best].fitness /= 2
			
			for i in Agents:
				i.fitness = 0
			
	TotalFitness = 0
	for i in range(len(Agents)):
		
		#This is where the inputs are
		if len(Food) <= 0:
			for i in range(5):
				AddFood()
		try:
			InputFOOD = Food[ClosestFood(Agents[i].posx, Agents[i].posy, Food)]
		except:
			print len(Food)
		
		dist = CalcDistance(InputFOOD, [Agents[i].posx, Agents[i].posy])
		Agents[i].Input[0] = dist/1526
		Agents[i].Input[1] = Normalise([InputFOOD[0] - Agents[i].posx, InputFOOD[1] - Agents[i].posy])[0]
		Agents[i].Input[4] = Normalise([InputFOOD[0] - Agents[i].posx, InputFOOD[1] - Agents[i].posy])[1]
		Agents[i].Input[2] = Agents[i].LookAtx
		Agents[i].Input[3] = Agents[i].LookAty		
	
		counter = -1
		
		#Neural network computations
		
		for j in range (len(Agents[i].Hidden)):
			for k in range (len(Agents[i].Input)):
				counter += 1

				Agents[i].Hidden[j] += Agents[i].Input[k] * Agents[i].WeightsIH[(counter)]
			Agents[i].Hidden[j] -= Agents[i].WeightBIAS1
			Agents[i].Hidden[j] = Sigmoid(Agents[i].Hidden[j])
	
		counter = -1
		for j in range(len(Agents[i].Output)):
			for k in range(len(Agents[i].Hidden)):
				counter += 1
				Agents[i].Output[j] += Agents[i].Hidden[k] * Agents[i].WeightsHO[counter]
			Agents[i].Output[j] -= Agents[i].WeightBIAS2			
			Agents[i].Output[j] = Sigmoid(Agents[i].Output[j])	
		TotalFitness += Agents[i].fitness		
		Agents[i].Clock += 1
		
		if Agents[i].Clock % 5 == 0:
			Agents[i].fitness -= 1
			Agents[i].fitness = Clamp(Agents[i].fitness, 1, 100000000)
	
		#Actually moving the agents

		lTrack = Agents[i].Output[0] #How much it will move to the left
		rTrack = Agents[i].Output[1] #How much it will move to the right
		
		RotForce = lTrack - rTrack 
		RotForce = Clamp(RotForce, -1, 1)
		
		Agents[i].Rotation += RotForce
		
		Speed = Agents[i].Output[2] * 5
		
		Agents[i].LookAtx = -math.sin((Agents[i].Rotation))
		Agents[i].LookAty = math.cos((Agents[i].Rotation)) 
		Agents[i].posx += Agents[i].LookAtx * Speed
		Agents[i].posy += Agents[i].LookAty * Speed
				
		FoodToDelete = []
		
		#Checks if an agent has touched food, if it has, then remove the food and give the agent some health
		
		for k in range (len(Food)):
			if CalcDistance([Agents[i].posx, Agents[i].posy], Food[k]) < 13:
				FoodToDelete.append(k) #Deleting the food
				Agents[i].Health += 1000 #Adding health 
				Agents[i].Health = Clamp(Agents[i].Health, -100, 4000)
				Agents[i].FoodColl += 1 #Adding to the food collected counter
				Agents[i].fitness += 100 #Increasing the fitness
		for i in range (len(FoodToDelete)):
			del Food[FoodToDelete[i] - i] 
	
		#If the agent moves out of the screen, just bring him in on the opposite side, kinda like pac-man?
		
		if Agents[i].posx > SCREENBOUNDX:
			Agents[i].posx = 0
		
		if (Agents[i].posx < 0):
			Agents[i].posx = SCREENBOUNDX
		
		if Agents[i].posy > SCREENBOUNDY:
			Agents[i].posy = 0
		
		if (Agents[i].posy < 0):
			Agents[i].posy = SCREENBOUNDY
		
		if Update:
			toUpdate.append(screen.fill(BACKGROUND))
			#pygame.display.flip()
				
		BestAgent = -99999999
		BestAgentIndex = 0
		
	if Update:
		for j in range(len(Food)):
			pygame.draw.circle(screen, Food[j][2], (int (Food[j][0]), int(Food[j][1])), 5)
			
	for j in range(len(Agents)):
		if Agents[j].FoodColl > MostFood:
			print "New Best!", Agents[j].FoodColl
		if Agents[j].FoodColl > BestAgent:
			BestAgent = Agents[j].FoodColl
			BestAgentIndex = j
			if BestAgent > MostFood:
				MostFood = BestAgent
		
		AgentsToDelete = []
	
	#Displaying the neural network
	counter = -1
	
	Select = Clamp(Select, 0, len(Agents))
	
	for i in range(len(Agents[Select].Input)):
		for j in range(len(Agents[Select].Hidden)):
			counter += 1
			pygame.draw.line(screen, (0,0,0), (975 + i * 50 , 700), (1025 + j * 50 , 650), int(Agents[Select].WeightsIH[counter]) + 1)
	counter = -1
	for i in range(len(Agents[0].Hidden)):
		for j in range(len(Agents[0].Output)):
			counter += 1
			pygame.draw.line(screen, (0,0,0), (1025 + i * 50, 650), (1025 + j * 50, 600), int(Agents[Select].WeightsHO[counter]) + 1)
	
	pygame.draw.circle(screen, (Clamp(Agents[Select].Input[0] * 500, 0, 255), 0, 0), (975, 700), 10)
	pygame.draw.circle(screen, (Clamp(Agents[Select].Input[1] * 500, 0, 255), 0, 0), (1025, 700), 10)
	pygame.draw.circle(screen, (Clamp(Agents[Select].Input[2] * 500, 0, 255), 0, 0), (1075, 700), 10)
	pygame.draw.circle(screen, (Clamp(Agents[Select].Input[3] * 500, 0, 255), 0, 0), (1125, 700), 10)
	pygame.draw.circle(screen, (Clamp(Agents[Select].Input[4] * 500, 0, 255), 0, 0), (1175, 700), 10)
	pygame.draw.circle(screen, (Clamp(Agents[Select].Hidden[0] * 500, 0, 255), 0, 0), (1025, 650), 10)
	pygame.draw.circle(screen, (Clamp(Agents[Select].Hidden[1] * 500, 0, 255), 0, 0), (1075, 650), 10)
	pygame.draw.circle(screen, (Clamp(Agents[Select].Hidden[2] * 500, 0, 255), 0, 0), (1125, 650), 10)
	pygame.draw.circle(screen, (Clamp(Agents[Select].Output[0] * 200, 0, 255), 0, 0), (1025, 600), 10)
	pygame.draw.circle(screen, (Clamp(Agents[Select].Output[1] * 200, 0, 255), 0, 0), (1075, 600), 10)
	pygame.draw.circle(screen, (Clamp(Agents[Select].Output[2] * 300, 0, 255), 0, 0), (1125, 600), 10)
	
	for i in range(len(Agents)):
		if Agents[i].Color == Agents[Select].Color and (SpeciesSelect == True):
			pygame.draw.line(screen, (255, 0, 0), (Agents[i].posx, Agents[i].posy), (Agents[Select].posx, Agents[Select].posy), 1)
	
	for i in range(len(Agents)):
		if Update:
			pygame.draw.line(screen, (255,0,0), (Agents[i].posx, Agents[i].posy), (Agents[i].posx + Agents[i].LookAtx * 20, Agents[i].posy + Agents[i].LookAty * 20))

			if i == Select:
				toUpdate.append(pygame.draw.circle(screen, (255,0,0), (int(Agents[i].posx/1), int(Agents[i].posy/1)), 9, 1))
				toUpdate.append(pygame.draw.circle(screen, (Agents[i].Color), (int(Agents[i].posx/1), int(Agents[i].posy/1)), 8))
				toUpdate.append(pygame.draw.circle(screen, (0,0,0), (int(Agents[i].posx) + 20, int(Agents[i].posy)), 2))
			else:
				toUpdate.append(pygame.draw.circle(screen, (0,0,255), (int(Agents[i].posx/1), int(Agents[i].posy/1)), 8, 1))
				toUpdate.append(pygame.draw.circle(screen, (Agents[i].Color), (int(Agents[i].posx/1), int(Agents[i].posy/1)), 7))
		
		Agents[i].Health -= 4
			
		if Agents[i].FoodColl >= 6: #If the agent collects more than 6 pieces of food, it asexually reproduces
			Replicate(Agents[i])
			Agents[i].FoodColl = 0
				
		if Agents[i].Health <= 0: 
			AgentsToDelete.append(i)
			TotalFitness -= Agents[i].fitness 
			
	for i in range(len(AgentsToDelete)):
		del Agents[AgentsToDelete[i] - i]
		if len(Agents) == 0:
			for i in range(5):
				Add()
		if Select > len(Agents):
			Select = len(Agents) - 1
		if Select < len(Agents):
			Select = 0


			
	if WhenToUpdate > 0:	
		if Ticks % WhenToUpdate == 0: 
			if Update:
				pygame.display.update(toUpdate)
				toUpdate = []
