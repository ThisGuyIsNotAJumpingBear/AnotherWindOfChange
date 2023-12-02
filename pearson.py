import csv

def load_annotator_labels():
    labels = {}
    with open("data/annotator.tsv") as fd:
        rd = csv.reader(fd, delimiter=" ")
        for row in rd:
            labels[row[0]] = row[1]

    return labels


def pearson_sgns():
    labels = load_annotator_labels()


if __name__ == "__main__":
    print(load_annotator_labels())