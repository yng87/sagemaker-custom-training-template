import polars as pl


def main(param: int):
    print("Hello from trainer!")
    print(f"param: {param}")
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    print(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=int, required=True)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
