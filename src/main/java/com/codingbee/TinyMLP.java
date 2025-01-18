//```
package com.codingbee;

//Importing libraries
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TinyMLP {
    //Declaring variables
    String networkPath;
    boolean[][] firstWeights;
    int[][][] weights;
    int[][] biases;
    int     h0,
            h1,
            h2;
    int inputLayerSize;
    int outputLayerSize;
    int     t11,
            t12,
            t21,
            t22,
            t31,
            t32;

    //The constructor, takes in path to the stored values
    //TODO: Remove path (because values will be loaded in some much more simple way)
    TinyMLP(String path){
        //Setting variable values, takes in path to saved values
        networkPath = path;
        inputLayerSize = 784;
        outputLayerSize = 10;
        h0 = 20;
        h1 = 10;
        h2 = 10;
        t11 = 70;
        t12 = 140;
        t21 = 35;
        t22 = 50;
        t31 = 29;
        t32 = 41;

        //Allocation of space
        firstWeights = new boolean[h0][inputLayerSize];
        weights = new int[3][][];
        weights[0] = new int[h1][h0];
        weights[1] = new int[h2][h1];
        weights[2] = new int[outputLayerSize][h2];
        biases = new int[4][];
        biases[0] = new int[h0];
        biases[1] = new int[h1];
        biases[2] = new int[h2];
        biases[3] = new int[outputLayerSize];

        //Initialization from files
        init();
    }

    //Initialization from files, will be done somehow more simple, based on the features of the pc
    //TODO: Super high level stuff, too much encapsulation, simply feed the values in or encode them to source code
    private void init(){
        // Load firstWeights
        File firstWeightsFile = new File(networkPath + "\\firstWeights.txt");
        if (firstWeightsFile.exists()) {
            List<boolean[]> firstWeightsList = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(firstWeightsFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] values = line.trim().split(" ");
                    boolean[] row = new boolean[values.length];
                    for (int i = 0; i < values.length; i++) {
                        row[i] = values[i].equals("1");
                    }
                    firstWeightsList.add(row);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            firstWeights = firstWeightsList.toArray(new boolean[0][]);
        }

        // Load weights
        File dir = new File(networkPath);
        if (dir.exists() && dir.isDirectory()) {
            File[] weightFiles = dir.listFiles((_, name) -> name.startsWith("weights_") && name.endsWith(".txt"));
            if (weightFiles != null) {
                weights = new int[weightFiles.length][][];
                for (File weightFile : weightFiles) {
                    int index = Integer.parseInt(weightFile.getName().replace("weights_", "").replace(".txt", ""));
                    List<int[]> weightsList = new ArrayList<>();
                    try (BufferedReader reader = new BufferedReader(new FileReader(weightFile))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            String[] values = line.trim().split(" ");
                            int[] row = new int[values.length];
                            for (int i = 0; i < values.length; i++) {
                                row[i] = Integer.parseInt(values[i]);
                            }
                            weightsList.add(row);
                        }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    weights[index] = weightsList.toArray(new int[0][]);
                }
            }
        }

        // Load biases
        if (new File(networkPath + "\\biases.txt").exists()) {
            List<int[]> biasesList = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(dir))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] values = line.trim().split(" ");
                    int[] row = new int[values.length];
                    for (int i = 0; i < values.length; i++) {
                        row[i] = Integer.parseInt(values[i]);
                    }
                    biasesList.add(row);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            biases = biasesList.toArray(new int[0][]);
        }
    }

    //Processing the pixels into probability list
    //TODO: Only issue are arrays
    public int[] processAsValues(boolean[] input){
        int[] hiddenLayer1 = new int[h0];
        for (int i = 0; i < h0; i++) {
            for (int j = 0; j < inputLayerSize; j++) {
                if (input[j] && firstWeights[i][j]) {
                    hiddenLayer1[i] += 2;
                }
                if (hiddenLayer1[i] > t12){
                    break;
                }
            }
            hiddenLayer1[i] -= biases[0][i];
        }
        norm(hiddenLayer1,t11, t12);


        int[] hiddenLayer2 = new int[h1];
        for (int i = 0; i < h1; i++) {
            for (int j = 0; j < h0; j++) {
                hiddenLayer2[i] += hiddenLayer1[j] * weights[0][i][j];
                if (hiddenLayer2[i] > t22){
                    break;
                }
            }
            hiddenLayer2[i] -= biases[1][i];
        }
        norm(hiddenLayer2, t21, t22);


        int[] hiddenLayer3 = new int[h2];
        for (int i = 0; i < h2; i++) {
            for (int j = 0; j < h1; j++) {
                hiddenLayer3[i] += hiddenLayer2[j] * weights[1][i][j];
                if (hiddenLayer3[i] > t32){
                    break;
                }
            }
            hiddenLayer3[i] -= biases[2][i];
        }
        norm(hiddenLayer3, t31, t32);


        int[] output = new int[outputLayerSize];
        for (int i = 0; i < outputLayerSize; i++) {
            for (int j = 0; j < h2; j++) {
                output[i] += hiddenLayer2[j] * weights[2][i][j];
                if (output[i] > 220){
                    break;
                }
            }
            output[i] -= biases[3][i];
        }

        return output;
    }

    //Helping method for normalizing the values into smaller ones
    //TODO: Only issue are arrays + potentially could be written all to one method
    private void norm(int[] activations, int t1, int t2){
        for (int i = 0; i < activations.length; i++) {
            if (activations[i] <= t1){
                activations[i] = 0;
                continue;
            }
            if (activations[i] > t1 && activations[i] <= t2){
                activations[i] = 1;
                continue;
            }
            if (activations[i] > t2) activations[i] = 2;
        }
    }

    public static void main(String[] args) {
        //Creates new virtual model
        TinyMLP tiny = new TinyMLP("src/main/resources/network");

        //Send 784 booleans (pixels, true = drawn there) and get ten values (representing "probability of given value being on picture)
        int[] response = tiny.processAsValues(new boolean[]{false, true, false, true, true, true, true, true, false, true, false, false, false, true, false, false, false, true, false, false, false, true, false, true, true, true, true, false, false, false, true, false, false, true, true, true, true, true, false, true, false, false, true, false, true, true, false, true, false, false, false, false, true, false, true, true, true, true, false, true, true, true, true, false, true, false, true, true, true, false, false, false, true, false, true, false, false, false, true, true, true, false, true, false, true, true, true, true, false, true, true, true, true, false, false, true, true, true, true, true, true, false, false, true, true, false, true, false, false, true, true, false, false, false, false, false, true, false, true, false, false, true, false, false, false, false, false, false, false, false, true, true, true, true, false, false, false, false, false, true, true, true, true, true, true, false, true, true, true, true, false, false, false, true, true, true, true, true, true, false, false, true, true, false, true, false, true, true, true, true, true, false, false, true, true, false, true, false, true, false, true, false, true, true, false, false, false, true, false, false, true, false, false, true, false, false, true, true, false, true, true, false, true, true, false, false, false, true, false, true, true, false, true, true, false, true, false, false, true, true, false, true, true, true, true, true, true, true, true, true, false, true, true, false, false, false, false, false, false, true, true, false, true, false, false, true, true, true, true, false, true, false, true, false, true, false, true, false, false, false, false, true, false, false, true, false, true, true, false, false, false, false, false, true, true, false, false, true, true, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, false, true, true, true, false, true, true, false, false, false, true, false, true, true, false, false, true, false, true, false, true, true, false, true, false, false, false, true, false, false, false, false, false, true, false, true, true, false, false, true, false, false, true, false, false, true, true, true, false, false, true, true, false, false, false, true, false, false, true, false, true, false, false, false, false, false, true, true, true, false, false, false, true, false, false, true, true, true, true, true, false, false, false, true, true, true, false, true, true, false, false, true, false, true, false, false, false, true, false, false, false, true, false, true, false, false, false, false, false, false, false, true, false, false, false, true, true, true, false, true, false, true, true, true, true, false, false, false, true, false, true, true, false, false, false, false, true, true, false, false, true, false, false, false, true, false, true, true, true, false, false, false, false, true, false, false, false, false, true, false, true, false, false, true, false, false, true, true, false, true, false, true, true, true, false, true, false, true, true, true, true, true, true, true, false, false, true, false, true, true, false, false, false, true, true, true, false, true, false, false, false, false, false, false, false, false, true, false, true, false, true, true, true, true, false, false, true, false, true, true, true, true, false, false, false, false, true, true, true, false, true, false, true, false, true, false, false, true, false, true, false, false, true, false, false, true, false, false, false, false, true, false, true, false, true, true, false, true, false, true, false, false, true, false, false, false, false, false, false, true, true, false, false, false, true, false, true, true, true, false, false, false, true, false, true, false, true, true, true, true, true, true, true, false, false, false, true, true, false, false, false, true, false, false, true, false, false, true, true, false, false, true, false, false, true, false, false, true, true, false, false, true, true, false, true, false, false, true, false, false, false, true, true, true, false, true, false, false, false, false, true, true, false, false, false, false, false, true, true, true, false, true, false, true, true, true, false, false, true, true, false, true, false, true, false, true, true, true, false, false, false, true, true, true, false, true, false, true, true, true, false, false, true, false, false, true, false, false, true, true, false, false, true, true, false, true, true, true, true, false, false, true, false, false, true, true, false, true, false, true, false, false, false, false, true, false, false, false, false, true, false, false, false, true, false, false, true, true, true, true, false, true, true, true, false, false, true, true, true, false, true, true, false, false, true, false, false, false, true, false, false, false, false, true, true, true, false, true, true, false, true, true, true, true, false, false, true, true, true, false, true, false, false, true, true, true, true, true, true, true, true, false, true, true,});
        for (int j : response) {
            System.out.println(j);
        }
    }
}