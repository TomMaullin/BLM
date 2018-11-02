    
def manyArgs(*arg):
    print(arg[1])
    print(len(arg))

with open('/home/tommaullin/Documents/DistOLS/DistOLS-py/binputs/Y0.txt') as a:

    y_files = []
    i = 0
    for line in a.readlines():

        print(repr(line))

        y_files.append(line.replace('\n', ''))

manyArgs('blaH', 'FHIR')    
