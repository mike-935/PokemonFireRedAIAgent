
class Move:
    def __init__(self, name, power, accuracy, type_id, effect_id, move_location):
        self.name = name
        self.power = power
        self.accuracy = accuracy
        self.effect_id = effect_id
        self.type_id = type_id
        self.move_location = move_location

    def to_tensor(self):
        pass

    def __str__(self):
        return (f"{self.name} (TypeID: {self.type_id}, Power: {self.power}, "
                f"Accuracy: {self.accuracy}, EffectID: {self.effect_id})")


