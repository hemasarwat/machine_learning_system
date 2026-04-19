class vehicle:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def get_info(self):
        return f"{self.year} {self.make} {self.model}"
    
class bicycle(vehicle):
    def __init__(self, make, model, year, type):
        super().__init__(make, model, year)
        self.type = type

    def get_info(self):
        return f"{super().get_info()} - {self.type}" 


def test_bicycle():
    bike = bicycle("cobra", "toyota", 2020, "mountain")
    assert hasattr(bike, "model")

def test_vehicle():
    car = vehicle("honda", "civic", 2019)
    assert hasattr(car, "make")
    assert hasattr(car, "model")

if __name__ == "__main__":
    test_bicycle()
    test_vehicle()
    print("All tests passed!")