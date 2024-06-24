import csv

def write_csv(mylst):
    import csv

    fields = [
        "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", 
        "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10",
        "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15", 
        "x16", "y16", "x17", "y17", "x18", "y18", "x19", "y19", "x20", "y20", 
        "x21", "y21", "lable"
    ]

    filename = "small_dataset.csv"

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(fields)
        writer.writerows(mylst)

data = []

for i in range(10):
    dataset = "Leo" + str(i) + ".csv"
    with open(dataset, 'r', newline='') as csvfile:
            heading = next(csvfile)
            data = data + list(csv.reader(csvfile))

write_csv(data)


