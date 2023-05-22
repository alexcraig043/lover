from lover import Lover
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

# Compute the total error for each threshold
def run_simulation(attractionDist, num_population, num_rounds, threshold, plot = False):
    # initializations
    attraction = []
    lovers = []
    married = set()
    POPULATION = num_population
    ROUNDS = num_rounds
    THRESHOLD = None
    
    if attractionDist == "Uniform":
        THRESHOLD = scipy.stats.uniform.ppf(threshold, 0, 1)
    elif attractionDist == "Normal":
        THRESHOLD = scipy.stats.norm.ppf(threshold, 0, 1)
    elif attractionDist == "Exponential":
        THRESHOLD = scipy.stats.expon.ppf(threshold, scale = 1)
    
    for _ in range(POPULATION):
        if attractionDist == "Uniform":
            # Sample a random attraction distribution for each person
            random_dist = np.random.uniform(0, 1, POPULATION)
            
            # Scale the attraction distribution to be between 0 and 1
            # random_dist = (random_dist - np.min(random_dist)) / (np.max(random_dist) - np.min(random_dist))
            
            # Append the attraction distribution to the list of attraction distributions
            attraction.append(random_dist)
            
        elif attractionDist == "Normal":
            # Sample a random attraction distribution for each person
            random_dist = np.random.normal(0, 1, POPULATION)
            
            # Scale the attraction distribution to be between 0 and 1
            # random_dist = (random_dist - np.min(random_dist)) / (np.max(random_dist) - np.min(random_dist))
            
            # Append the attraction distribution to the list of attraction distributions
            attraction.append(random_dist)
            
        elif attractionDist == "Exponential":
            # Sample a random attraction distribution for each person
            random_dist = np.random.exponential(1, POPULATION)
            
            # Scale the attraction distribution to be between 0 and 1
            # random_dist = random_dist / max(random_dist)
            
            # Append the attraction distribution to the list of attraction distributions
            attraction.append(random_dist)
            
    for i in range(POPULATION):
        lover_attraction = attraction[i].copy()
        lover = Lover(i, [], False, lover_attraction)
        lovers.append(lover)
        
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
        plot_simulation(num_married, ROUNDS, POPULATION, attractionDist, threshold)
    
    # print(num_married)
    # print("Number of single people: " + str(len(singles)))
    # print("Number of married people: " + str(len(married)))
    return num_married

# Plot the number of married people per round
def plot_simulation(num_married, ROUNDS, POPULATION, attractionDist, threshold):
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

    title = "Number of Married People Per Round for Threshold = " + str(threshold) + " and " + attractionDist + " Attraction Distribution"
    plt.title(title)
    file_name = "./src/figures/num_married_" + attractionDist + "_" + str(threshold) + ".png"
    fig.savefig(file_name)
    plt.close()

# run_simulation("Exponential", 100, 100, .75, True)
# run_simulation("Uniform", 100, 100, .75, True)

# Compute the average number of married people for each threshold
def aggregate_simulations(attractionDist, num_population, num_rounds, num_simulations, plot, read_csv = False, write_csv = False):
    # Make array of thresholds from 0 to 1 in increments of 0.01
    thresholds = np.arange(0, 1.01, 0.01)
    
    # Instantiate an array to store the average number of married people for each threshold
    avg_num_married = []
    
    # Read the csv file if read_csv is True
    if (read_csv):
        file_name = "./src/data/avg_num_married_" + attractionDist + ".csv"
        avg_num_married = pd.read_csv(file_name, header=None).values.flatten().tolist()
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
        
    # Plot the average number of married people per round for each threshold
    if (plot):
        plot_agg_simulations(avg_num_married, thresholds, num_rounds, attractionDist)
        # Compute the total error
        compute_total_error(avg_num_married, thresholds, num_population, attractionDist)
        
    # Write the avg_num_married to a csv file
    if (write_csv):
        file_name = "./src/data/avg_num_married_" + attractionDist + ".csv"
        pd.DataFrame(avg_num_married).to_csv(file_name, index=False, header=False)
        
    return avg_num_married
        
# Plot the average number of married people per round for each threshold
def plot_agg_simulations(avg_num_married, thresholds, num_rounds, attractionDist):
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
    
    title = "Average Number of Married People Per Round for " + attractionDist + " Attraction Distribution"
    plt.title(title)
    file_name = "./src/figures/avg_num_married_" + attractionDist + ".png"
    fig.savefig(file_name)
    plt.close()
   
# Compute and plot the total error for each threshold
def compute_total_error(avg_num_married, thresholds, num_population, attractionDist, threshold_weight = 1, married_prop_weight = 1, plot = True):
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
        total_error.append(threshold_weight * threshold_error + married_prop_weight * married_prop_error)
        
    # Get minimum total error
    min_total_error = min(total_error)
    
    # Get threshold with minimum total error
    min_total_error_threshold = thresholds[total_error.index(min_total_error)]
    
    # Get maximum total error
    max_total_error = max(total_error)
    
    if (plot == True):
        plot_total_error(total_error, thresholds, min_total_error, min_total_error_threshold, max_total_error, attractionDist)
        
    return total_error

def plot_total_error(total_error, thresholds, min_total_error, min_total_error_threshold, max_total_error, attractionDist):
    # Plot the total error
        
    # Set ylim
    if (max_total_error > 1):
        ylim = max_total_error + .05
    else:
        ylim = 1
    
    x = thresholds
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    ylab = "Total error"
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel(ylab)
    ax1.step(x, total_error)
    ax1.tick_params(axis='y')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, ylim)
    ax1.grid(True)
    
    # Add a horizontal line at the minimum total error
    ax1.axhline(y=min_total_error, color='r', linestyle='--')
    
    # Add a vertical line at the threshold with the minimum total error
    ax1.axvline(x=min_total_error_threshold, color='r', linestyle='--')
    
    # Add a legend containing the minimum total error and the threshold with the minimum total error
    rounded_min_total_error = round(min_total_error, 4)
    rounded_threshold = round(min_total_error_threshold, 4)
    ax1.legend(['Total error', 'Minimum total error: ' + str(rounded_min_total_error), 'Threshold with minimum total error: ' + str(rounded_threshold)])
    
    # Adjust the layout
    fig.tight_layout()
    # Add some margin
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    title = "Total Error Per Threshold for " + attractionDist + " Attraction Distribution"
    plt.title(title)
    file_name = "./src/figures/total_error_" + attractionDist + ".png"
    fig.savefig(file_name)
    plt.close()

aggregate_simulations("Exponential", num_population = 100, num_rounds = 100, num_simulations = 25, plot = True, read_csv = False, write_csv = False)