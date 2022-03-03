import json
from activation_function import *
import numpy as np

"""
Fungsi untuk membaca file model dan mengisi nilai-nilainya ke variable bersesuaian
"""
def readFile(filePath: str):
    # Opening JSON file
    f = open(filePath)
    
    # returns JSON object as
    # a dictionary
    data = json.load(f) 
    
    # Closing file
    f.close()
    return data

def print_hidden_layer(hidden_layer):
    for i in range(len(hidden_layer)):
        print("Hidden Layer " + str(i))
        print("Activation Function: " + str(hidden_layer[i]["activation_function"]))
        print("Weight: " + str(hidden_layer[i]["weight"]))
        print("Bias: " + str(hidden_layer[i]["bias"]))
        print("")

def calcNet(inputMatrix, hiddenMatrix):
    return np.matmul(inputMatrix, hiddenMatrix.transpose())

def main():
    # fileModel = input("Masukan file model : ")
    # fileInput = input("Masukan file input : ")

    modelData = readFile("soalslide.json")
    inputData = readFile("input.json")

    hiddenLayers = modelData["hidden_layer"]
    outputLayer = modelData["output_layer"]

    # X input as matrix   
    for item in inputData["input"]:
        item.insert(0, 1) # Insert 1 for bias at every input intance
    
    
    # Convert to matrix hidden layer
    # adding bias to every neuron h
    for i in range (len(hiddenLayers[0]["bias"])):
        hiddenLayers[0]["weight"][i].insert(0, hiddenLayers[0]["bias"][i])
    

    # Convert to matrix output layer
    # adding bias to every neuron h
    for i in range (len(outputLayer["bias"])):
        outputLayer["weight"][i].insert(0, outputLayer["bias"][i])
    
    inputMatrix = np.matrix(inputData["input"])
    hiddenLayerMatrix = np.matrix(hiddenLayers[0]["weight"])
    outputLayerMatrix = np.matrix(outputLayer["weight"])
    print(inputMatrix)
    print(inputMatrix.shape)
    print(hiddenLayerMatrix)
    print(hiddenLayerMatrix.shape)

    print(calcNet(inputMatrix, hiddenLayerMatrix))
    # PSEUDOCODE, DON'T TOUCH
    # 1 orang urusin looping forwardnya di main (TODO: Urusin looping forward di main, Faris)
    # For each hidden layer : (TODO: Ngitung perkalian matriks pertama kali, Jafar)
    # Process each bias and weight using matrix mult --> 1 function
    # Compute h --> 1 function using activation function (TODO: Ngitung h pake activation function, Alwan)
    # return its result matrix untuk diproses di layer selanjutnya) (dont forget to add bias 1 ) (TODO: idem line sebelumnya)

    # Compute in output layer (TODO: Ngitung output layer buat perhitungan final, Alip)
    # Kalau di latihan uts itu bagian net_o (TODO: idem line sebelumnya)


    print(hiddenLayers)
    print(outputLayer)

if __name__=="__main__":
    main()