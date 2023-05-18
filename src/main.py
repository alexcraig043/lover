from lover import Lover
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Compute the total error for each threshold
def run_simulation(attractionDist, num_population, num_rounds, threshold, plot):
    # initializations
    attraction = []
    lovers = []
    married = set()
    THRESHOLD = threshold
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
    num_married.append(0)

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
        num_married.append(len(married)) # Add number of married people to array
        
    if (plot):
        plot_simulation(num_married, ROUNDS, POPULATION)
    
    # print(num_married)
    # print("Number of single people: " + str(len(singles)))
    # print("Number of married people: " + str(len(married)))
    return num_married

# Plot the number of married people per round
def plot_simulation(num_married, ROUNDS, POPULATION):
    # Plot the number of married people per round
    x = np.arange(0, ROUNDS + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)

    ax1.set_xlabel('Round')
    ax1.set_ylabel('Number of married people')
    ax1.step(x, num_married)
    ax1.tick_params(axis='y')
    ax1.set_xlim(0, ROUNDS)
    ax1.set_ylim(0, POPULATION)
    ax1.grid(True)
    
    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Proportion of married people')
    ax2.step(x, np.array(num_married) / POPULATION)
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 1)
    
    # Adjust the layout
    fig.tight_layout()  
    # Add some margin
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1) 

    plt.title('Number of Married People Per Round')
    fig.savefig('./src/figures/num_married.png')
    plt.close()

# run_simulation("uniform", 100, 100, 0.8, True)

# Compute the average number of married people for each threshold
def aggregate_simulations(attractionDist, num_population, num_rounds, num_simulations, plot, read_csv=False):
    # Make array of thresholds from 0 to 1 in increments of 0.01
    thresholds = np.arange(0, 1.01, 0.01)
    
    # Instantiate an array to store the average number of married people for each threshold
    avg_num_married = []
    
    # Read the csv file if read_csv is True
    if (read_csv):
        avg_num_married = pd.read_csv("./src/data/avg_num_married/" + attractionDist + ".csv", header=None).values.flatten().tolist()
    else: # Else, run the simulation
        # For each threshold
        for threshold in thresholds:
            print("Threshold: " + str(threshold))
            # Instantiate an array to store the number of married people for each simulation
            num_married_threshold = []
            
            # Run the simulation num_simulations times
            for _ in range(num_simulations):
                # Run the simulation
                num_married_threshold_sim = run_simulation(attractionDist, num_population, num_rounds, threshold, False)
                # Append the number of married people from the last round to the array
                num_married_threshold.append(num_married_threshold_sim[-1])
                
            # Find the average number of people married for threshold
            avg_num_married_threshold = np.mean(num_married_threshold)
            # Append the average number of people married to the array
            avg_num_married.append(avg_num_married_threshold)
        
    if (plot):
        plot_agg_simulations(avg_num_married, thresholds, num_rounds)
        compute_total_error(avg_num_married, thresholds, num_population)
        
    # Write the avg_num_married to a csv file
    pd.DataFrame(avg_num_married).to_csv("./src/data/avg_num_married/" + attractionDist + ".csv", index=False, header=False)
        
    return avg_num_married
        
    
# Plot the average number of married people per round for each threshold
def plot_agg_simulations(avg_num_married, thresholds, num_rounds):
    # Plot the number of married people per round
    x = thresholds
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    ylab = "Average number of married people by round" + str(num_rounds)
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel(ylab)
    ax1.step(x, avg_num_married)
    ax1.tick_params(axis='y')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 100)
    ax1.grid(True)
    
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Proportion of married people')
    ax2.step(x, np.array(avg_num_married) / 100)
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 1)
    
    
    # Adjust the layout
    fig.tight_layout()
    # Add some margin
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.title('Average Number of Married People Per Threshold')
    fig.savefig('./src/figures/avg_num_married.png')
    plt.close()
   
# Compute and plot the total error for each threshold
def compute_total_error(avg_num_married, thresholds, num_population):
    # Instantiate an array to store the total error for each threshold
    total_error = []
    
    # For each threshold
    for i in range(len(thresholds)):
        # Threshold error
        threshold_error = 1 - thresholds[i]
        
        # Proportion of people married at threshold
        married_prop = avg_num_married[i] / num_population
        # Married proportion error
        married_prop_error = 1 - married_prop
        
        # Total error
        total_error.append(threshold_error + married_prop_error)
        
    # Get minimum total error
    min_total_error = min(total_error)
    
    # Plot the total error
    x = thresholds
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    ylab = "Total error"
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel(ylab)
    ax1.step(x, total_error)
    ax1.tick_params(axis='y')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    
    # Add a horizontal line at the minimum total error
    ax1.axhline(y=min_total_error, color='r', linestyle='--')
    
    # Add a vertical line at the threshold with the minimum total error
    ax1.axvline(x=thresholds[total_error.index(min_total_error)], color='r', linestyle='--')
    
    # Add a legend containing the minimum total error and the threshold with the minimum total error
    rounded_min_total_error = round(min_total_error, 4)
    rounded_threshold = round(thresholds[total_error.index(min_total_error)], 4)
    ax1.legend(['Total error', 'Minimum total error: ' + str(rounded_min_total_error), 'Threshold with minimum total error: ' + str(rounded_threshold)])
    
    # Adjust the layout
    fig.tight_layout()
    # Add some margin
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.title('Total Error Per Threshold')
    fig.savefig('./src/figures/total_error.png')
    plt.close()
        
    return total_error

aggregate_simulations("uniform", 100, 100, 25, True)