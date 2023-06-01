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
        
    # Fill in the rest of the array with the last value
    while len(num_married) < num_rounds + 1:
        num_married.append(num_married[-1])
        
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

# run_simulation("Uniform", 100, 100, .35, True)

# Compute the average number of married people for each threshold
def aggregate_simulations(attractionDist, num_population, num_rounds, num_simulations, plot, round_to_plot, read_csv = False, write_csv = False):
    # Set the round to Plot
    round_to_plot = round_to_plot - 1
    
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
            print(attractionDist + " Threshold: " + str(threshold))
            # Instantiate an array to store the number of married people for each simulation
            num_married_threshold = []
            
            # Run the simulation num_simulations times
            for _ in range(num_simulations):
                # Run the simulation
                num_married_threshold_sim = run_simulation(attractionDist, num_population, num_rounds, threshold, False)
                # Append the number of married people for each round to the array
                num_married_threshold.append(num_married_threshold_sim)
                
            # Find the average number of people married for each round
            avg_num_married_threshold = np.mean(num_married_threshold, axis=0)
            
            # Append the average number of people married to the array
            avg_num_married.append(avg_num_married_threshold)
        
        # convert avg_num_married to 2d list
        avg_num_married = np.array(avg_num_married).tolist()
        
    # Plot the average number of married people per round for each threshold
    if (plot):
        plot_agg_simulations(avg_num_married, thresholds, num_population, num_rounds, attractionDist, round_to_plot)
        # Compute the total error
        compute_total_error(avg_num_married, thresholds, num_population, attractionDist, round_to_plot = round_to_plot)
        
    # Write the avg_num_married to a csv file
    if (write_csv):
        file_name = "./src/data/avg_num_married_" + attractionDist + ".csv"
        pd.DataFrame(avg_num_married).to_csv(file_name, index=False, header=False)
        
    return avg_num_married
        
# Plot the average number of married people per round for each threshold
def plot_agg_simulations(avg_num_married, thresholds, num_population, num_rounds, attractionDist, round_to_plot):
    # Get an array of the round_to_plot'th round for each threshold
    avg_num_married = [avg_num_married[i][round_to_plot] for i in range(len(thresholds))]
    
    # Plot the number of married people per round
    x = thresholds
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    ylab = "Average number of married people by round" + str(num_rounds)
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel(ylab)
    ax1.step(x, avg_num_married)
    ax1.tick_params(axis='y')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, num_population)
    ax1.grid(True)
    
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Proportion of married people')
    ax2.step(x, np.array(avg_num_married) / num_population)
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 1)
    
    
    # Adjust the layout
    fig.tight_layout()
    # Add some margin
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    title = "Average Number of Married People Per Threshold for " + attractionDist + " Attraction Distribution For Round " + str(round_to_plot + 1) + " of " + str(num_rounds) + " Rounds"
    plt.title(title)
    file_name = "./src/figures/avg_num_married_" + attractionDist + "_round_" + str(round_to_plot + 1) + ".png"
    fig.savefig(file_name)
    plt.close()
   
# Compute and plot the total error for each threshold
def compute_total_error(avg_num_married, thresholds, num_population, attractionDist, threshold_weight = 1, married_prop_weight = 1, plot = True, round_to_plot = -1):
    # Instantiate an array to store the total error for each threshold
    total_error = []
    
    # For each threshold
    for i in range(len(thresholds)):
        # Calculate the threshold error
        threshold_error = 1 - thresholds[i]
        
        # Weight the threshold error
        threshold_error *= threshold_weight
        
        # Define an array to store the total error for each round
        total_error_threshold = []
        
        # For each round
        for r in range(len(avg_num_married[i])):
            # Calculate the proportion of people married at threshold
            married_prop = avg_num_married[i][r] / num_population
            
            # Calculate the married proportion error
            married_prop_error = 1 - married_prop
            # Weight the married proportion error
            married_prop_error *= married_prop_weight
            
            # Calculate the total error
            total_error_threshold.append(threshold_error + married_prop_error)
        
        # Add the total error for each round to the array
        total_error.append(total_error_threshold)
    
    # Define an array of length num_rounds to store the minimum total error for each round
    min_total_error = [-1] * len(avg_num_married[0])
    
    # Define an array of length num_rounds to store the maximum total error for each round
    max_total_error = [-1] * len(avg_num_married[0])
    
    # Define an array of length num_rounds to store the threshold with the minimum total error for each round
    min_total_error_threshold = [-1] * len(avg_num_married[0])
    
    for i in range(len(total_error)):
        for r in range(len(total_error[i])):
            if (min_total_error[r] == -1 or total_error[i][r] < min_total_error[r]):
                min_total_error[r] = total_error[i][r]
                min_total_error_threshold[r] = thresholds[i]
            if (max_total_error[r] == -1 or total_error[i][r] > max_total_error[r]):
                max_total_error[r] = total_error[i][r]
                
    if (plot == True):
        plot_total_error(total_error, thresholds, min_total_error, min_total_error_threshold, max_total_error, attractionDist, round_to_plot)
        plot_optimal_thresholds(min_total_error_threshold, attractionDist)
        
    return min_total_error_threshold
        
def plot_total_error(total_error, thresholds, min_total_error, min_total_error_threshold, max_total_error, attractionDist, round_to_plot = -1):
    # Get an array of the round_to_plot'th round for each total error
    total_error = [total_error[i][round_to_plot] for i in range(len(thresholds))]
    
    min_total_error = min_total_error[round_to_plot]
    min_total_error_threshold = min_total_error_threshold[round_to_plot]
    max_total_error = max_total_error[round_to_plot]

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
    
    title = "Total Error Per Threshold for " + attractionDist + " Attraction Distribution" + " For Round " + str(round_to_plot + 1) + " of " + str(len(total_error) - 1) + " Rounds"
    plt.title(title)
    file_name = "./src/figures/total_error_" + attractionDist + "_round_" + str(round_to_plot + 1) + ".png"
    fig.savefig(file_name)
    plt.close()

def plot_optimal_thresholds(min_total_error_threshold, attractionDist):
    # Set x to be rounds going from 1 to len(min_total_error_threshold)
    x = [i + 1 for i in range(len(min_total_error_threshold))]
    x.pop(0)
    
    # Set y to be the threshold with the minimum total error for each round
    y = min_total_error_threshold
    y.pop(0)
    
    # Set ylim
    ylim = 1
    
    # Plot the threshold with the minimum total error for each round
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Optimal Threshold" )
    ax1.step(x, y)
    ax1.tick_params(axis='y')
    ax1.set_xlim(0, len(min_total_error_threshold))
    ax1.set_ylim(0, ylim)
    ax1.grid(True)
    
    # Adjust the layout
    fig.tight_layout()
    # Add some margin
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    title = "Optimal Threshold Per Round for " + attractionDist + " Attraction Distribution"
    plt.title(title)
    file_name = "./src/figures/optimal_threshold_" + attractionDist + ".png"
    fig.savefig(file_name)
    plt.close()

aggregate_simulations("Uniform", num_population = 100, num_rounds = 100, num_simulations = 100, plot = True, round_to_plot = 100, read_csv = False, write_csv = False)