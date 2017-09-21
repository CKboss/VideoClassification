import os

ROOT = "/workspace/mmdata/"


def main():
    folders = os.listdir(ROOT)
    for f in folders:
        for file in os.listdir(os.path.join(ROOT, f)):
            parts = file.split("_")
            if len(parts) == 4:
                for i in range(2, 3):
                    parts[i] = complete(parts[i])

            elif len(parts) == 5:
                for i in range(2, 4):
                    parts[i] = complete(parts[i])
            ends = parts[-1].split(".")
            parts[-1] = complete(ends[0]) + ".jpg"
            new_file = "_".join(parts)

            os.rename(file, os.path.join(ROOT, new_file))


def complete(num_str):
    if len(num_str) == 1:
        return "000" + num_str
    elif len(num_str) == 2:
        return "00" + num_str
    elif len(num_str) == 3:
        return "0" + num_str
    else:
        return num_str


def test_complete():
    files = ["lsvc019567.flv_flow_19_9_10.jpg", "lsvc019567.flv_frame_14_16.jpg"]

    for file in files:
        parts = file.split("_")
        if len(parts) == 4:
            for i in range(2, 3):
                parts[i] = complete(parts[i])

        elif len(parts) == 5:
            for i in range(2, 4):
                parts[i] = complete(parts[i])
        ends = parts[-1].split(".")
        parts[-1] = complete(ends[0]) + ".jpg"
        new_file = "_".join(parts)
        print(new_file)


if __name__ == "__main__":
    test_complete()
    # main()
