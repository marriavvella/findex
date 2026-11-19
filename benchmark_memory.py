
import sys
import array


def main():
    values = list(range(10000))
    compact = array.array("I", values)

    list_size = (
        sys.getsizeof(values)
        + sum(sys.getsizeof(x) for x in values)
    )

    array_size = sys.getsizeof(compact)

    saving = (1 - array_size / list_size) * 100

    print("Memory benchmark")
    print("----------------")
    print(f"Number of integers: {len(values)}")
    print(f"Python list: {list_size} bytes")
    print(f"Compact array: {array_size} bytes")
    print(f"Memory saving: {saving:.2f}%")


if __name__ == "__main__":
    main()
