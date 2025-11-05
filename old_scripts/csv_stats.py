import csv
import matplotlib.pyplot as plt


def read_csv(filename):
    try:
        with open(filename, 'r', newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=';')
            data = list(csv_reader)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def read_column_from_data(data, column_name):
    try:
        column_data = [row[column_name] for row in data]
        return column_data
    except KeyError:
        print(f"Error: Column '{column_name}' not found")
        return None

def to_floats(values):
    nums = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s == '':
            continue
        try:
            nums.append(float(s))
        except ValueError:
            continue
    return nums
"""
if __name__ == "__main__":
    data = read_csv("zadanie1-data.csv")
    if data:
        #print(data[0:10])
        # Get age data
        age_data = read_column_from_data(data, 18)
        yes=0
        no=0
        unknown=0
        none=0;
        if age_data:

            for row in age_data:
                if row=='yes':
                    yes+=1
                elif row=='no':
                    no+=1
                elif row=='unknown':
                    unknown+=1
                elif row=='':
                    none+=1
            print(f"yes: {yes}, no: {no}, unknown: {unknown}, none: {none}")
            print("Total:", yes+no+unknown+none)
"""

if __name__ == "__main__":
    data = read_csv("zadanie1-data.csv")
    column_data = read_column_from_data(data, 19)
    numeric = to_floats(column_data[1:])
    #print(column_data[1:])



    plt.figure(figsize=(20, 30))
    plt.violinplot(numeric)
    plt.title('Box Plot of Column 19')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()
