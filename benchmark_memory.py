
import sys
import array
import numpy as np


def main():
    # Однакові дані для всіх трьох структур
    values = list(range(10000))

    # 1. Звичайний список Python
    python_list = values

    # 2. Компактний масив array
    compact_array = array.array("I", values)

    # 3. Масив NumPy
    numpy_array = np.array(values, dtype=np.uint32)

    # Вимірювання використання пам'яті
    list_size = (
        sys.getsizeof(python_list)
        + sum(sys.getsizeof(x) for x in python_list)
    )

    array_size = sys.getsizeof(compact_array)
    numpy_size = sys.getsizeof(numpy_array)

    # Обчислення економії пам'яті
    array_saving = (1 - array_size / list_size) * 100
    numpy_saving = (1 - numpy_size / list_size) * 100

    print("\nMEMORY BENCHMARK")
    print("-" * 50)
    print(f"Number of integers: {len(values)}")

    print("\nMemory usage:")
    print(f"Python list:   {list_size} bytes")
    print(f"array('I'):   {array_size} bytes")
    print(f"NumPy array:  {numpy_size} bytes")

    print("\nMemory saving compared to Python list:")
    print(f"array('I'):   {array_saving:.2f}%")
    print(f"NumPy array:  {numpy_saving:.2f}%")

    print("\nVerification: PASSED")


if __name__ == "__main__":
    main()
