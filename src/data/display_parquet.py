from sys import argv
import pandas as pd

def main() -> None:
    data = pd.read_parquet(argv[1])    
    print(data)


if __name__ == "__main__":
    main()