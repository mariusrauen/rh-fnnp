@dataclass
class Info:
    name: str
    abbreviation: str
    process_description: str
    mainflow: str
    location: str
    exact_location: str
    capacity: str
    unit_per_year: str | float

    exact_location = 'unknown'

    def find_main_product():
        pass

    def find_location() :
        pass

    def find_process_description():
        pass

def get_info():
    r"Product: (.*?),"
    pass

