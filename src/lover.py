# Sabin Hart
# Create the Lover class
# April 30, 2023

class Lover:
    def __init__(self, ID, history, marriage_status, attraction):
        self.ID = int(ID)
        self.history = history
        self.marriage_status = bool(marriage_status)
        self.attraction = attraction

    def __str__(self):
        if self.marriage_status:
            return str(self.ID) + ", married"
        else:
            return str(self.ID) + ", single"

    def get_married(self):
        return self.marriage_status

    def get_id_num(self):
        return self.ID

    def get_history(self):
        return self.history

    def get_attraction_list(self):
        return self.attraction

