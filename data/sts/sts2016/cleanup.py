
def main():
    inputs_file = input()


    with open(inputs_file, 'r'), open('clean' + inputs_file, 'w') as inf, outf:

        lines = inf.getlines()

        for line in lines:

            outf.write('\n')
