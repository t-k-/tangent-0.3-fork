__author__ = 'mauricio'

import sys

def main():
    if len(sys.argv) < 2:
        print("Usage")
        print("\tpython extract_unique_expressions.py [input_files]")
        print("")
        print("Where:")
        print("\tinput_files\t: Index files to process")
        return

    full_dict = {}
    doc_id = -1
    for filename in sys.argv[1:]:
        in_file = open(filename, "r")
        all_lines = in_file.readlines()
        in_file.close()

        for line in all_lines:
            if line[0] == "E":
                parts = line.strip().split("\t")

                expression = parts[1]

                if not expression in full_dict:
                    full_dict[expression] = True

                    subparts = parts[2].split(",")
                    first_location = subparts[0][1:]
                    if len(subparts) == 1:
                        first_location = first_location[:-1]


                    print(expression + "\t" + str(doc_id) + "\t" + str(first_location))

            if line[0] == "D":
                parts = line.strip().split("\t")

                doc_id = int(parts[1])


main()