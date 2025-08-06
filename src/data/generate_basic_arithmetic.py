import random
from datasets import Dataset, DatasetDict, load_dataset
import json

def randomize_template(op, a, b):
    templates = {
        "add": [
            f"what is {a} + {b}?",
            f"add {a} and {b}",
            f"{a} plus {b} equals?",
        ],
        "sub": [
            f"subtract {b} from {a}",
            f"what is {a} minus {b}?",
        ],
        "mul": [
            f"what is {a} times {b}?",
            f"multiply {a} and {b}",
        ],
       
    }
    return random.choice(templates[op])

def randomize_answer_template(op, a, b):
    templates = {
        "add": [
            f"{a+b}",
            f"{a} + {b} = {a+b}",
            f"Adding {a} and {b} gives {a + b}.",
            f"The result of adding {a} and {b} is {a + b}.",
            f"{a} plus {b} equals {a + b}."
        ],
        "sub": [
            f"{a - b}",
            f"{a} - {b} = {a-b}",
            f"Subtracting {b} from {a} gives {a - b}.",
            f"{a} minus {b} is {a - b}",
            f"The result of subtracting {b} from {a} is {a - b}."
        ],
        "mul": [
            f"{a * b}",
            f"Multiply {a} and {b}, and you get {a * b}.",
            f"{a} multiplied by {b} equals {a * b}.",
            f"The product of {a} and {b} is {a * b}."
        ],
       
    }
    return random.choice(templates[op])


def generate_example():
    ops = ["add", "sub", "mul"]
    op = random.choice(ops)
    a = random.randint(0, 500)
    b = random.randint(0, 500)

    if op == "add":
        input_text = randomize_template(op,a,b)
        answer = randomize_answer_template(op,a,b)
    elif op == "sub":
        input_text = randomize_template(op,a,b)
        answer = randomize_answer_template(op,a,b)
    elif op == "mul":
        input_text = randomize_template(op,a,b)
        answer = randomize_answer_template(op,a,b)
        
    return {"input": input_text, "output": answer}



class ArithmeticDataGenerator():
    def __init__(self,num):
        super().__init__()
        self.num = num
        
    def generate(self):
        data = [generate_example() for _ in range(self.num)]
        with open("arithmetic_dataset.json", "w", encoding="utf-8") as f:
            for example in data:
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")
        dataset = Dataset.from_list(data)
        dataset_dict = DatasetDict({"train": dataset})

        return dataset_dict
    
    def load(self,dir=None):
        dataset = load_dataset("json", data_files="arithmetic_dataset.json", split="train")
        dataset_dict = DatasetDict({"train": dataset})

        return dataset_dict