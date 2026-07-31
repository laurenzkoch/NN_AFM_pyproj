


def valid_y_values(x: int, max_y: int) -> list[int]:
    if x <= 0:
        raise ValueError("x muss positiv sein.")

    power_of_three = 1

    while x % (power_of_three * 3) == 0:
        power_of_three *= 3

    if power_of_three == 1:
        return []

    return [
        y
        for y in range(1, max_y + 1)
        if y % power_of_three != 0
    ]


def main() -> None:
    pass


if __name__ == "__main__":
    main()