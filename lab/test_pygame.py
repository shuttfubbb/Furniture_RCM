
from enum import Enum

class FurnitureType(Enum):
    CARBINET = 0
    TEACHER_TABLE = 1
    ANOTHER = 2

print(FurnitureType.ANOTHER.value)