from dataclasses import dataclass
@dataclass
class Stream:
    name: str
    cost: float
    cost_unit: str
    amount: float
    amount_unit: str
    cost_per_kg: float
    class_: int = 1           #why 1?



#@dataclass
#class costs:



#@dataclass 
#class info:
    