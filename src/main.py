from lover import Lover
import random
import numpy as np
import matplotlib.pyplot as plt

def main(attractionDist, num_population, num_rounds):
    # initializations
    attraction = []
    lovers = []
    married = set()
    THRESHOLD = 0.7
    POPULATION = num_population
    ROUNDS = num_rounds

    for _ in range(POPULATION):
        if attractionDist == "uniform":
            attraction.append(np.random.uniform(0, 1, POPULATION))
        elif attractionDist == "normal":
            attraction.append(np.random.normal(0.5, 0.1, POPULATION))
        elif attractionDist == "exponential":
            attraction.append(np.random.exponential(0.5, POPULATION))
            
    # Flatten attraction list
    attraction = np.array(attraction).flatten()
    
    # Order the attraction list
    attraction.sort()
    
    for i in range(POPULATION):
        random.shuffle(attraction)
        love = Lover(i, [], False, attraction.copy())
        lovers.append(love)
        
    # Create a list of single individuals (all individuals single at start)
    singles = lovers.copy()
    
    # Create an array for number of married people each round
    num_married = []

    # ROUNDS
    for _ in range(ROUNDS):
        # Safety check, if less than 2 singles, break the loop
        if len(singles) < 2:
            break

        paired_singles = []
        
        # Loop through each single person
        while len(singles) >= 2:
            lover1, lover2 = random.sample(singles, 2)

            # Check if the mutual attraction is above the threshold
            if lover1.attraction[lover2.ID] > THRESHOLD and lover2.attraction[lover1.ID] > THRESHOLD:
                lover1.marriage_status = True
                lover2.marriage_status = True
                married.add(lover1)
                married.add(lover2)
                singles.remove(lover1)
                singles.remove(lover2)
            else:
                paired_singles.append(lover1)
                paired_singles.append(lover2)
                singles.remove(lover1)
                singles.remove(lover2)

        singles = paired_singles.copy()  # Reassign remaining singles for next round
        num_married.append(len(married))

    print(num_married)
    print("Number of single people: " + str(len(singles)))
    print("Number of married people: " + str(len(married)))



main("uniform", 100, 100)
