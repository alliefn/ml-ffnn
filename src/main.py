import json

"""
Fungsi untuk membaca file model dan mengisi nilai-nilainya ke variable bersesuaian
"""
def readModel(filePath: str):
    # Opening JSON file
    f = open(filePath)
    
    # returns JSON object as
    # a dictionary
    data = json.load(f) 
    
    # Closing file
    f.close()
    return data

def activation_function(name_of_the_func):
    match name_of_the_func:
        case 'sigmoid':
            # sigmoid algorithm
            print("Algoritma sigmoid placeholder")
            return 0
        case 'linear':
            print("linear")
            return 0
        case 'relu':
            print("relu")
            return 0
        case 'softmax':
            print("softmax") 
            return 0
        case _:
            print("Unknown activation function")
            return 0

def main():
    fileModel = input("Masukan file model : ")
    fileInput = input("Masukan file input : ")

    # Create parser, which reads the input file
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="model file", type=str, default=fileModel)
    parser.add_argument("-i", "--input", help="input file", type=str, default=fileInput)
    parser.add_argument("-o", "--output", help="output file", type=str, default="output.txt")

    modelData = readModel(fileModel)
    hiddenLayers = modelData["hidden_layer"]
    outputLayer = modelData["output_layer"]

    for i in range(len(hiddenLayers)):
        print("Hidden Layer " + str(i))
        print("Activation Function: " + str(hiddenLayers[i]["activation_function"]))
        print("Weight: " + str(hiddenLayers[i]["weight"]))
        print("Bias: " + str(hiddenLayers[i]["bias"]))
        print("")

    print(hiddenLayers)
    print(outputLayer)
    print("main")

if __name__=="__main__":
    main()