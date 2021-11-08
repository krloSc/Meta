import re
import os
def get_parameters(file_name):
    try:
        path=os.getcwd()
        file=open(path+"/Metaheuristics/parameters/"+file_name+".param",'r')
        lst=file.read().split('\n')
        parameters = dict()
        for lines in lst:
            parameter =  re.search("\w*", lines).group()
            value =re.search('([0-9]|-).*',lines).group()
            if value.find(".") != -1: #if value has a decimal point
                value = float(value)
            else:
                value = int(value)
            parameters[parameter] = value

    except Exception as e:
        print(file_name+" - Parameters file not found - Using default values")
        print(e)
    return parameters

if __name__ == "__main__":
    get_parameters("prueba")
