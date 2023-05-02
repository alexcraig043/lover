# Sabin, Ryan, Gary, Ellie, Davis, Alex
# Last Chances Project
# April 30, 2023

from lover import Lover
import random


def main():
    # initializations
    attraction = []
    lovers = []
    single = set()
    married = set()
    THRESHOLD = 0.7
    POPULATION = 100
    ROUNDS = 500

    for i in range(1, POPULATION + 1):
        attraction.append(i / 100)

    for i in range(0, POPULATION):
        random.shuffle(attraction)
        clone = attraction.copy()
        love = Lover(i, [], False, clone)
        lovers.append(love)
        single.add(love)

    runs = 0
    while True:
        still_single = set()
        still_single.clear()

        for i in range(1, (len(single) // 2) + 1):
            lover1 = single.pop()
            id1 = lover1.id_num()

            lover2 = single.pop()
            id2 = lover2.id_num()

            lover1.history.append(id1)
            lover2.history.append(id2)

            if (
                lover1.attraction_list()[id2] > THRESHOLD
                and lover2.attraction_list()[id1] > THRESHOLD
            ):
                married.add(lover1)
                married.add(lover2)
            else:
                still_single.add(lover1)
                still_single.add(lover2)

        # print("Single: " + str(single))
        # print("Married: " + str(married))
        # print("Still Single: " + str(still_single))
        single = still_single.copy()
        runs += 1
        if runs > ROUNDS:
            break
    print(len(married))


main()
