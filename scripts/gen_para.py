import argparse
from decimal import Decimal, getcontext

def generate_lines():
    getcontext().prec = 10
    
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--n", 
        type=str,
        default="3"
    )
    parser.add_argument(
        "--u", 
        type=str,
        default="3"
    )
    parser.add_argument(
        "--st", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--ed", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--it", 
        type=str, 
        required=True
    )
    
    args = parser.parse_args()

    # print("5 5 5 1.00", end="\n")
    # print("10 10 10 1.00", end="\n")
    # print("25 25 25 1.00", end="\n")
    try:
        st = Decimal(args.st)
        ed = Decimal(args.ed)
        it = Decimal(args.it)
        n = int(args.n)
        u = int(args.u)

        if it <= 0:
            print("Error: Interval (it) must be positive.")
            return

        current_val = ed
        while current_val >= st:
            for i in range(u):
                print(current_val, end = " ")
            for i in range(n-u):
                print(-1, end = " ")
            print("1.00", end="\n")
            current_val -= it        
        # current_val = st
        # while current_val <= ed:
        #     for i in range(n):
        #         print(current_val, end = " ")
        #     print("1.00", end="\n")
        #     current_val += it

    except Exception as e:
        print(f"Error: Invalid input. {e}")
    for j in [25, 10, 5]:
        for i in range(u):
            print(j, end = " ")
        for i in range(n-u):
            print(-1, end = " ")
        print("1.00", end="\n")
    # print("25 25 25 1.00", end="\n")
    # print("10 10 10 1.00", end="\n")
    # print("5 5 5 1.00", end="\n")

if __name__ == "__main__":
    generate_lines()