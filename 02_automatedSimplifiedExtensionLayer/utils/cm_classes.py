from dataclasses import dataclass
@dataclass
class Stream:
    name: tuple[str]
    cost: tuple[float]
    cost_unit: tuple[str]
    amount: tuple[float]
    amount_unit: tuple[str]
    cost_per_kg: tuple[float]
    class_: tuple[int] = 1 



#@dataclass
#class costs:



#@dataclass 
#class info:
    